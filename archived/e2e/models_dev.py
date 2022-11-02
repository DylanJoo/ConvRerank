import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from typing import Optional, Union, Dict, Any
from loss import InBatchKLLoss, InBatchNegativeCELoss, PairwiseCELoss

class ColBertForLICQE(BertPreTrainedModel):
    """ColBert for Late-interacted Conversational Query Embeddings.
    This class provides:
    (1) Train like ColBert efficiently (In-batch negative Loss)
    (2) TCT training (In-bathc negative Loss + KL Loss)
    """
    def __init__(self, config, **kwargs):

        super(ColBertForLICQE, self).__init__(config)

        self.dim = kwargs.pop('dim', 128)
        self.gamma = kwargs.pop('gamma', 0.1)
        self.temperature = kwargs.pop('temperature', 0.25)
        self.similarity_metric = kwargs.pop('similarity_metric', 'cosine')

        # Colbert variants
        self.kd_teacher = kwargs.pop('kd_teacher', None)
        self.colbert_type = kwargs.pop('colbert_type', 'colbert')
        self.skiplist = {}

        if kwargs.pop('mask_punctuation', True):
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.skiplist = {
                    w: True for symbol in string.punctuation for w in \
                            [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]
            }

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, self.dim, bias=False)

        self.init_weights()

    def forward(self,
                u_input_ids: Optional[torch.Tensor] = None,
                u_attention_mask: Optional[torch.Tensor] = None,
                u_token_type_ids: Optional[torch.Tensor] = None,
                c_input_ids: Optional[torch.Tensor] = None,
                c_attention_mask: Optional[torch.Tensor] = None,
                c_token_type_ids: Optional[torch.Tensor] = None,
                d_input_ids: Optional[torch.Tensor] = None,
                d_attention_mask: Optional[torch.Tensor] = None,
                d_token_type_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                keep_d_dims: bool = True,
                **kwargs):

        # Context cqe
        ## Context reader
        c = self.bert(
            input_ids=c_input_ids,
            attention_mask=c_attention_mask,
            token_type_ids=c_token_type_ids,
            **kwargs
        ) # B Lc H

        ## Question reader
        u = self.bert(
            input_ids=u_input_ids,
            attention_mask=u_attention_mask,
            token_type_ids=u_token_type_ids,
            **kwargs
        ) # B Lq H

        ## Document reader
        d = self.bert(
            input_ids=d_input_ids,
            attention_mask=d_attention_mask,
            token_type_ids=d_token_type_ids,
            **kwargs
        )
        d_mask = self.mask(d_input_ids)

        # ColBert 
        if self.colbert_type == 'colbert-inbatch':
            C = self.colbert_pooler(c.last_hidden_state) # (B Lc H)
            U = self.colbert_pooler(u.last_hidden_state) # (B Lu H)
            D = self.colbert_pooler(d.last_hidden_state, mask=d_mask, keep_dims=keep_d_dims)
            Q = torch.cat((C, U), 1) # (B Lq H)

            scores_context_inbatch = self.inbatch_score(C, U) # (B B)
            loss_context = InBatchNegativeCELoss(scores_context_inbatch)
            scores_inbatch = self.inbatch_score(Q, D) # (B 2*B)
            loss = InBatchNegativeCELoss(scores_inbatch)
            return {'score': scores_inbatch, 
                    'context_score': scores_context_inbatch,
                    'loss': 1*loss + 1*loss_context,}

        # TctColbert
        if self.colbert_type == 'tctcolbert':
            ## TctColBert: Knowledge distillation loss (in-batch KL divergence)
            ### student
            C = self.avg_pooler(c.last_hidden_state)
            U = self.avg_pooler(u.last_hidden_state)
            D = self.avg_pooler(d.last_hidden_state)
            Q = C + U

            scores_inbatch_s = Q @ D.permute(1, 0) # (B 2*B)
            scores_context_inbatch_s = C @ U.permute(1, 0) # (B 2*B)
            ### teacher
            with torch.no_grad():
                scores_t = self.kd_teacher.forward(
                    u_input_ids,
                    u_attention_mask, 
                    u_token_type_ids,
                    c_input_ids,
                    c_attention_mask, 
                    c_token_type_ids,
                    d_input_ids,
                    d_attention_mask,
                    d_token_type_ids,
                    **kwargs
                )
            loss = InBatchNegativeCELoss(scores_inbatch_s)
            loss_context = InBatchNegativeCELoss(scores_context_inbatch_s)
            loss_kl = InBatchKLLoss(scores_inbatch_s, scores_t['score'], self.temperature)
            loss_context_kl = InBatchKLLoss(scores_context_inbatch_s, scores_t['context_score'], self.temperature)

            return {'score': scores_inbatch_s, 
                    'context_score': scores_context_inbatch_s, 
                    'loss': self.gamma*(loss+loss_context) + (1-self.gamma)*(loss_kl+loss_context_kl)}

    def inbatch_score(self, Q, D):
        Q_prime = Q.view(-1, Q.size(-1)) # (B*Lq H)
        D_prime = D.view(-1, D.size(-1)) # (B*2*Ld H)
        B, Lq, Lh = Q.size(0), Q.size(1), D.size(1)
        print(Q_prime.size(), D_prime.size())

        # if self.similarity_metric == 'cosine':
        return (Q_prime @ D_prime.permute(1, 0)).view(B, Lq, -1, Lh).permute(0, 2, 1, 3).max(-1).values.sum(-1) #(B 2B Lq Ld) -> (B 2B Lq) -> (B 2B)

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] \
                for d in input_ids.cpu().tolist()]
        mask = torch.tensor(mask, device=self.device).unsqueeze(2).float()
        return mask  # B L 1

    def colbert_pooler(self, tokens_last_hidden, mask=1, keep_dims=True):
        X = self.linear(tokens_last_hidden)
        X = X * mask # for d
        X = F.normalize(X, p=2, dim=2)

        if not keep_dims:  # for d
            X, mask = X.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            X = [d[mask[idx]] for idx, d in enumerate(X)]
        return X
    
    def avg_pooler(self, tokens_last_hidden): # (B Lq H) -> (B H)
        return torch.mean(tokens_last_hidden[:, 4:, :], dim=-2)

