#Train/Inference on TPU with T5X

0. Inititate TPU-VM and Google VM 
See the [instructions]() frrom google for detail.

1. In this document, we use the T5X for example, which is built by JAX.
First install the following requirement.

- Activate TPU-VM, you will enter the TPU-VM
```
TPU_NAME=<YOUR TPU-VM NAME> 
ZONE=<YOUR TPU REGION> (e.g. us-central1-f)
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE}
```

- Install JAX for T5X
```
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html~
git clone --branch=main https://github.com/google-research/t5x
cd t5x
python3 -m pip install -e '.[tpu]' -f \
          https://storage.googleapis.com/jax-releases/libtpu_releases.html

# the permission will be denied in the begginning
gcloud auth application-default login
```

2. Modified the 't5x/t5x/models.py

- Replace the 'score_batch()' functions, and change the output scores with logits (of true/false)
```
# (Option1) you can directly change the t5x repo's contents by
cp models.py t5x/t5x/models.py
```
Since T5X is constantly updating, you can accordinly refine the codes like
```python

  def score_batch(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
      first_logits: bool = False,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, Any]]]:
    """Compute log likelihood score on a batch."""
    weights = batch['decoder_loss_weights']
    target_tokens = batch['decoder_target_tokens']

    if return_intermediates:
      logits, modified_variables = self._compute_logits(
          params=params, batch=batch, mutable=['intermediates'])

      # Inside self.module, we called nn.Module.sow to track various
      # intermediate values. We extract them here.
      intermediates = flax_core.unfreeze(
          modified_variables.get('intermediates', {}))

      # Track per-token labels and loss weights as well. These are not
      # intermediate values of logit computation, so we manually add them here.
      intermediates.setdefault('decoder', {})
      intermediates['decoder']['target_tokens'] = (target_tokens,)
      intermediates['decoder']['loss_weights'] = (weights,)
      # Note that the values are singleton tuples. This is because values inside
      # `intermediates` should be tuples tracking all instantiations of a value.
      # These values each have just one instantiation, hence singletons.
    else:
      logits = self._compute_logits(params, batch)  # type: jnp.ndarray

    # Purposefully don't use config.z_loss because that term is for training
    # stability and shouldn't affect our reported scores.

    ######### ######### Change the logits output ######### #########
    if first_logits:
        sequence_scores = logits[:, 0:1, target_tokens[0, :]] # true/false tokens

    ######### ######### ######### #########

    token_scores = -losses.cross_entropy_with_logits(
        logits,
        common_utils.onehot(
            target_tokens[:, 0:1], logits.shape[-1], on_value=1, off_value=0),
        z_loss=0.0)[0] * weights

    sequence_scores = token_scores.sum(-1)

    if return_intermediates:
      return sequence_scores, intermediates

    return sequence_scores


```

