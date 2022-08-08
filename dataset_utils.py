import multiprocessing
from datasets import concatenate_datasets

def prepare_for_monot5(dataset):

    def qp_pointwise_ranking(x, label='true'):
        if label == 'true':
            x['passage'] = x.pop('pos_passage')
            x['label'] = label
        else:
            x['passage'] = x.pop('neg_passage')
            x['label'] = label
        return x

    ## Prepare pointwise ranking dataset
    dataset_pos = dataset.map(
            qp_pointwise_ranking,
            fn_kwargs={"label": 'true'},
            remove_columns=['neg_passage'],
            num_proc=multiprocessing.cpu_count(),
    )
    dataset_neg = dataset.map(
            qp_pointwise_ranking,
            fn_kwargs={"label": 'false'},
            remove_columns=['pos_passage'],
            num_proc=multiprocessing.cpu_count(),
    )

    return concatenate_datasets([dataset_pos, dataset_neg])
