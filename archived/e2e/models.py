"""
This codes modified from original ColBert Paper's repositary:

[ColBert](https://github.com/stanford-futuredata/ColBERT/blob/master/colbert/modeling/colbert.py)
"""
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from typing import Optional, Union, Dict, Any
from loss import InBatchKLLoss, InBatchNegativeCELoss, PairwiseCELoss

class ColBertForCQE(BertPreTrainedModel):
    """ColBert for Conversational Query Embeddings 
    This class provides:
    (1) Train like ColBert (pairwise CE Loss)
    (2) Train like ColBert efficiently (In-batch negative Loss)
    (3) TCT training (In-bathc negative Loss + KL Loss)
    """
    def __init__(self, config, **kwargs):
                 # query_maxlen, 
                 # doc_maxlen, 
                 # mask_punctuation, 
                 # dim=128, 
                 # teacher=None,
                 # similarity_metric='cosine'):

        super(ColBertForCQE, self).__init__(config)

        # self.query_maxlen = kwargs.pop('query_maxlen', 32)
        # self.doc_maxlen = kwargs.pop('doc_maxlen', 128)
        self.dim = kwargs.pop('dim', 128)
        self.gamma = kwargs.pop('gamma', 0.1)
        self.temperature = kwargs.pop('temperature', 0.25)
        # use for knowledge distilation
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
                q_input_ids: Optional[torch.Tensor] = None,
                q_attention_mask: Optional[torch.Tensor] = None,
                q_token_type_ids: Optional[torch.Tensor] = None,
                d_input_ids: Optional[torch.Tensor] = None,
                d_attention_mask: Optional[torch.Tensor] = None,
                d_token_type_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                keep_d_dims: bool = True,
                **kwargs):
        """
        In this TctColBert model, 
        we use the shared bert model as Document & Query's encoder
        
        Note that colbert used pairwise ranking loss,
        so the q_input_ids would be repetitve at first dimension, 
        while d_input_ids would concatenate by pos and neg document
        """

        q = self.bert(
            input_ids=q_input_ids,
            attention_mask=q_attention_mask,
            token_type_ids=q_token_type_ids,
            **kwargs
        )

        d = self.bert(
            input_ids=d_input_ids,
            attention_mask=d_attention_mask,
            token_type_ids=d_token_type_ids,
            **kwargs
        )
        d_mask = self.mask(d_input_ids)

        # ColBert 
        if self.colbert_type == 'colbert':
            Q = self.colbert_pooler(q.last_hidden_state)
            D = self.colbert_pooler(d.last_hidden_state, mask=d_mask, keep_dims=keep_d_dims)

            ## ColBert stanrd loss (pairwise loss)
            scores_colbert = self.pairwise_score(Q, D)  # (2*B 1)
            loss = PairwiseCELoss(scores_colbert)
            return {'score': scores_colbert, 'loss': loss}

        if self.colbert_type == 'colbert-inbatch':
            Q = self.colbert_pooler(q.last_hidden_state)
            D = self.colbert_pooler(d.last_hidden_state, mask=d_mask, keep_dims=keep_d_dims)

            ## ColBert improved loss (in-batch CE loss)
            scores_inbatch = self.inbatch_score(Q, D) # (B 2*B) # in batch CE loss
            loss = InBatchNegativeCELoss(scores_inbatch)
            return {'score': scores_inbatch, 'loss': loss}

        # TctColbert
        if self.colbert_type == 'tctcolbert':
            ## TctColBert: Knowledge distillation loss (in-batch KL divergence)
            ### student
            Q = self.avg_pooler(q.last_hidden_state)
            D = self.avg_pooler(d.last_hidden_state)
            scores_inbatch_s = Q @ D.permute(1, 0) # (B 2*B)
            ### teacher
            with torch.no_grad():
                scores_inbatch_t = self.kd_teacher.forward(
                    q_input_ids,
                    q_attention_mask, 
                    q_token_type_ids,
                    d_input_ids,
                    d_attention_mask,
                    d_token_type_ids,
                    **kwargs
                )['score']
            loss = InBatchNegativeCELoss(scores_inbatch_s)
            loss_kl = InBatchKLLoss(scores_inbatch_s, scores_inbatch_t, self.temperature)
            return {'score': scores_inbatch_s, 'loss': self.gamma*loss + (1-self.gamma)*loss_kl}

    def pairwise_score(self, Q, D):
        """ Max sim operator for pairwise loss

        1. tokens cos-sim: (B Lq H) X (B H Ld) = (B Lq Ld)
        2. max token-token cos-sim: (B Lq), the last dim indicates max of qd-cos-sim of q
        3. sum by batch: (B 1)
        """
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)
    
    def inbatch_score(self, Q, D):
        Q_prime = Q.view(-1, Q.size(-1)) # (B*Lq H)
        D_prime = D.view(-1, D.size(-1)) # (B*2*Ld H)
        B, Lq, Lh = Q.size(0), Q.size(1), D.size(1)

        # if self.similarity_metric == 'cosine':
        return (Q_prime @ D_prime.permute(1, 0)).view(B, Lq, B*2, Lh).permute(0, 2, 1, 3).max(-1).values.sum(-1) #(B 2B Lq Ld) -> (B 2B Lq) -> (B 2B)

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
        # embeddings = tokens_last_hidden
        # return np.average(embeddings[:, 4:, :], axis=-2) # B H

