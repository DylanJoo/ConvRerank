import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import (
        PaddingStrategy,
        PreTrainedTokenizerBase
)

@dataclass
class DataCollatorFormonoT5:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True
    istrain: Union[bool] = False
    # Spec
    # query_maxlen: Optional[int] = None 
    # doc_maxlen: Optional[int] = None
    # context_maxlen: Optional[int] = None
    # utterance_maxlen: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # input
        text_inputs = [f"Query: {batch['query']} Document: {batch['passage']} Relevant:" \
                for batch in features]
        inputs = self.tokenizer(
                text_inputs,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors=self.return_tensors
        )


        # qid-pid pairs 
        ids = [(batch['qid'], batch['pid']) for batch in features]

        # labeling (if training)
        if self.istrain:
            # labels
            targets = self.tokenizer(
                    [text['label'] for text in features],
                    truncation=True,
                    return_tensors=self.return_tensors
            ).input_ids
            inputs['labels'] = target

        return inputs, ids

@dataclass
class PointwiseConvDataCollatorForT5:
    tokenizer: PreTrainedTokenizerBase
    # context_maxlen: Optional[int] = None
    # utterance_maxlen: Optional[int] = None
    query_maxlen: Optional[int] = None
    doc_maxlen: Optional[int] = None
    return_tensors: Optional[str] = None
    num_history: Optional[int] = 1
    num_history_utterances: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        u_texts = [f"{text['utterance']}" for text in features]
        d_texts = [f"{text['passage']}" for text in features] 
        c_texts = []

        for i, text in enumerate(features):
            c_text = text['context'].split('|') # odds are queries, even are docs
            # c_texts.append('|'.join(c_text[-self.num_history_utterances:]))
            if self.num_history_utterances is not None:
                cq_text = [c for i, c in enumerate(c_text) \
                        if i % 2 == 0][-self.num_history_utterances]
                c_texts.append('|'.join(cq_text))
            else:
                c_texts.append('|'.join(c_text[-self.num_history*2:]))

        ## Document tokenization
        d_inputs = self.tokenizer(
                [d + " Relevant:" for d in d_texts],
                max_length=self.doc_maxlen,
                padding="longest",
                truncation="longest_first",
                return_tensors=self.return_tensors
        )

        ## Utterance text + Context text + (truncated) Document listOfTokens
        inputs = self.tokenizer(
                [f"Query: {u} Context: {c} " for (u, c) in zip(u_texts, c_texts)],
                max_length=self.query_maxlen,
                padding="max_length",
                truncation="longest_first",
                add_special_tokens=False,
                return_tensors=self.return_tensors
        )

        ## target text
        targets = self.tokenizer(
                [text['label'] for text in features],
                truncation=True,
                return_tensors=self.return_tensors
        )

        # Concatentate
        for k in ['input_ids', 'attention_mask']:
            inputs[k] = torch.cat((inputs[k], d_inputs[k]), 1)

        # labels
        inputs['labels'] = targets.input_ids

        return inputs

# # monoT5
# @dataclass
# class monoT5DataCollator:
#     tokenizer: PreTrainedTokenizerBase
#     padding: Union[bool, str, PaddingStrategy] = True
#     truncation: Union[bool, str] = True
#     max_length: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#     return_tensors: str = "pt"
#     padding: Union[bool, str] = True
#
#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
#
#         batch_text = self.tokenizer(
#             [v['text'] for v in features],
#             padding=self.padding,
#             truncation=self.truncation,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors=self.return_tensors,
#         )
#
#         batch_id = [(v['qid'], v['docid']) for v in features]
#
#         return batch_text, batch_id
#
# # convmonot5
# @dataclass
# class ConvmonoT5DataCollator:
#     tokenizer: PreTrainedTokenizerBase
#     padding: Union[bool, str, PaddingStrategy] = True
#     truncation: Union[bool, str] = True
#     max_length: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#     return_tensors: str = "pt"
#     padding: Union[bool, str] = True
#
#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
#
#         batch_text = self.tokenizer(
#             [v['text'] for v in features],
#             padding=self.padding,
#             truncation=self.truncation,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors=self.return_tensors,
#         )
#
#         batch_id = [(v['qid'], v['docid']) for v in features]
#
#         return batch_text, batch_id
