import functools
import seqio
import tensorflow_datasets as tfds
from t5.evaluation import metrics
from t5.data import preprocessors
import tensorflow.compat.v1 as tf

vocabulary = seqio.SentencePieceVocabulary(
        'gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model'
)
output_features = {
        'inputs': seqio.Feature(vocabulary=vocabulary),
        'targets': seqio.Feature(vocabulary=vocabulary),
}


def monot5_train_preprocessor(ds):

    def to_inputs_and_targets(ex):
        return {
                "inputs": tf.strings.join([ex["inputs"]]),
                "targets": tf.strings.join([ex["targets"]]),
        }
    return ds.map(
            to_inputs_and_targets,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

def monot5_eval_preprocessor(ds):

    def to_inputs_and_targets(ex):
        return {
                "inputs": tf.strings.join([ex["inputs"]]),
                "targets": tf.strings.join(['true false'] * len(ex) ) 
                # [Infer] we use truefalse tokens' logits
        }
    return ds.map(
            to_inputs_and_targets,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
    )


seqio.TaskRegistry.add(
        'torank_teacher',
        source=seqio.TextLineDataSource(
            {'test': 'gs://cnclab/cast/data/torank/cast20.canard.train.answer+rewrite.infer.top1000.tsv'}
        ),
        preprocessors=[
            functools.partial(
                preprocessors.parse_tsv,
                field_names=["inputs"]
            ),
            monot5_eval_preprocessor,
            seqio.preprocessors.tokenize_and_append_eos,
        ],
        metric_fns=[],
        output_features=output_features
)

seqio.TaskRegistry.add(
        'torank_student',
        source=seqio.TextLineDataSource(
            {'test': 'gs://cnclab/cast/data/torank/cast20.canard.train.rewrite.infer.top1000.tsv'}
        ),
        preprocessors=[
            functools.partial(
                preprocessors.parse_tsv,
                field_names=["inputs"]
            ),
            monot5_eval_preprocessor,
            seqio.preprocessors.tokenize_and_append_eos,
        ],
        metric_fns=[],
        output_features=output_features
)
