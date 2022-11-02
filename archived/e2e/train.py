"""
Codes for finetunning ColBert model

Includes following setups
- ColBert with pairwise loss (following instructions from original paper)
- COlBert with in-bathch negatives loss

Backbone models
- bert-base-uncased
"""
import sys
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    DefaultDataCollator,
)

from datasets import load_dataset, DatasetDict
from models import ColBertForCQE
from datacollator import IRTripletCollator

import os
os.environ["WANDB_DISABLED"] = "true"

# Arguments: (1) Model arguments (2) DataTraining arguments (3)
@dataclass
class OurModelArguments:
    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(default='bert-base-uncased')
    model_type: Optional[str] = field(default='bert-base-uncased')
    config_name: Optional[str] = field(default='bert-base-uncased')
    tokenizer_name: Optional[str] = field(default='bert-base-uncased')
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    # Cutomized arguments
    colbert_type: Optional[str] = field(default="colbert")
    dim: Optional[int] = field(default=128)
    kd_teacher_model_name_or_path: Optional[str] = field(default="castorini/tct_colbert-v2-hnp-msmarco-r2")
    # pooler_type: str = field(default="cls")
    # temp: float = field(default=0.05)

@dataclass
class OurDataArguments:
    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    train_file: Optional[str] = field(default="data/orconvqa/sample.jsonl")
    eval_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)
    # Customized arguments
    max_q_seq_length: Optional[int] = field(default=32)
    max_p_seq_length: Optional[int] = field(default=128)

@dataclass
class OurTrainingArguments(TrainingArguments):
    # Huggingface's original arguments. 
    output_dir: str = field(default='./models')
    seed: int = field(default=42)
    data_seed: int = field(default=None)
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    max_steps: int = field(default=100)
    save_steps: int = field(default=5000)
    eval_steps: int = field(default=2500)
    evaluation_strategy: Optional[str] = field(default='no')
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    weight_decay: float = field(default=0.0)
    logging_dir: Optional[str] = field(default='./logs')
    warmup_ratio: float = field(default=0.1)
    warmup_steps: int = field(default=0)
    resume_from_checkpoint: Optional[str] = field(default=None)
    learning_rate: float = field(default=5e-5)

def main():

    # Parseing argument for huggingface packages
    parser = HfArgumentParser((OurModelArguments, OurDataArguments, OurTrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_datalcasses()
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # [CONCERN] Deprecated? or any parser issue.
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # config and tokenizers
    config_kwargs = {
            "output_hidden_states": True
    }
    tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir, 
            "use_fast": model_args.use_fast_tokenizer
    }
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)

    # model 
    model_teacher = ColBertForCQE.from_pretrained(
            pretrained_model_name_or_path=model_args.kd_teacher_model_name_or_path,
            config=config,
            colbert_type='colbert-inbatch',
    ) if model_args.colbert_type == 'tctcolbert' else None

    model_kwargs = {
            'dim': model_args.dim,
            'similarity_metric': 'cosine', 
            'mask_punctuation': True,
            'kd_teacher': model_teacher, 
            'colbert_type': model_args.colbert_type
    }
    model = ColBertForCQE.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=config,
            **model_kwargs
    )

    # Dataset 
    ## Loading form json
    if training_args.do_eval:
        dataset = DatasetDict.from_json({
            "train": data_args.train_file,
            "eval": data_args.eval_file
        })
    else:
        dataset = DatasetDict.from_json({"train": data_args.train_file,})
        dataset['eval'] = None

    # data collator (transform the datset into the training mini-batch)
    ## Preprocessing
    triplet_collator = IRTripletCollator(
            tokenizer=tokenizer,
            query_maxlen=data_args.max_q_seq_length,
            doc_maxlen=data_args.max_p_seq_length,
            in_batch_negative=(model_args.colbert_type != 'colbert')
    )

    # Trainer
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['eval'],
            data_collator=triplet_collator
    )
    
    # ***** strat training *****
    results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == '__main__':
    main()
