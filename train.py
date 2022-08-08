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
from datacollator import PointwiseConvDataCollatorForT5
from models import monoT5
from dataset_utils import prepare_for_monot5

import os
os.environ["WANDB_DISABLED"] = "true"

# Arguments: (1) Model arguments (2) DataTraining arguments (3)
@dataclass
class OurModelArguments:
    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(default='t5-small')
    model_type: Optional[str] = field(default='t5-small')
    config_name: Optional[str] = field(default='t5-small')
    tokenizer_name: Optional[str] = field(default='t5-small')
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
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
    train_file: Optional[str] = field(default=None)
    eval_file: Optional[str] = field(default=None)
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
    remove_unused_columns: bool = field(default=False)

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
    model = monoT5.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=config,
    )

    # Dataset 
    ## Loading form json
    if training_args.do_eval:
        dataset = DatasetDict.from_json({
            "train": data_args.train_file,
            "eval": data_args.eval_file
        })
        dataset_train_mono = prepare_for_monot5(dataset['train'])
        dataset_eval_mono = prepare_for_monot5(dataset['eval'])
    else:
        dataset = DatasetDict.from_json({"train": data_args.train_file,})
        dataset_train_mono = prepare_for_monot5(dataset['train'])


    # data collator (transform the datset into the training mini-batch)
    ## Preprocessing
    conv_datacollator = PointwiseConvDataCollatorForT5(
            tokenizer=tokenizer,
            query_maxlen=data_args.max_q_seq_length,
            doc_maxlen=data_args.max_p_seq_length,
            return_tensors='pt',
            num_history=3,
    )

    # Trainer
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_train_mono,
            eval_dataset=dataset_eval_mono if training_args.do_eval else None,
            data_collator=conv_datacollator
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
