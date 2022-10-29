import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration

class monoT5(T5ForConditionalGeneration):
    targeted_tokens = ['true', 'false']
    # tokenizer_name = 'google/t5-base'

    def set_tokenizer(self):
        if 'base' in self.name_or_path or self.name_or_path is None:
            self.tokenizer = T5Tokenizer.from_pretrained('google/t5-base')
        elif 'large' in self.name_or_path:
            self.tokenizer = T5Tokenizer.from_pretrained('google/t5-large')
        else:
            raise ValueError("Cannot identify tokenizer byname_or_path.")

    def set_targets(self, tokens=None):
        """
        Parameters:
            tokens: list of string
        """
        if tokens is None:
            tokens = self.targeted_tokens

        tokenized_tokens = self.tokenizer(tokens, add_special_tokens=False)
        self.targeted_ids = [x for xs in tokenized_tokens.input_ids for x in xs]
        # print(f"{len(tokens)} targeted tokens set")
        # print(list(zip(tokens, self.targeted_ids)))
        print("Ready for predict()")

    def predict(self, batch):
        # Perpare BOS labels

        softmax = nn.Softmax(dim=1)

        for k in batch:
            batch[k] = batch[k].to(self.device)

        dummy_labels = torch.full(
                batch.input_ids.size(), 
                self.config.decoder_start_token_id
        ).to(self.device)
        batch_logits = self.forward(**batch, labels=dummy_labels).logits
        return softmax(batch_logits[:, 0, self.targeted_ids]).detach().cpu().numpy() # B 2
