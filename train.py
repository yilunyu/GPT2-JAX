import jax
import optax

import utils

@jax.jit
def train_step(state, x, y):
  def _loss(params):
    predictions = state.apply_fn(params, x, training=True)
    loss = optax.softmax_cross_entropy_with_integer_labels(predictions, y)
    return loss.mean()
  loss, grads = jax.value_and_grad(_loss)(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss

@jax.jit
def eval_step(state, x, y):
  predictions = state.apply_fn(state.params, x, training=False)
  return optax.softmax_cross_entropy_with_integer_labels(predictions, y).mean()

# TODO: implement training loop with micro batching.
