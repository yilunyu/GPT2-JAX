import data_loader
import jax
import optax
import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm
import utils
import yaml

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

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

def run_training_loop(train_loader, eval_loader, config):
   
   for step in range(config.train_steps):
      x, y = train_loader.next_batch()
      # train_step(state, x, y)

# TODO: implement training loop with micro batching.

if __name__ == "__main__":
  # Open the YAML file
  with open('config.yaml', 'r') as file:
    # Load the YAML content
    config = yaml.safe_load(file)
  # fw = load_dataset("HuggingFaceFW/fineweb-edu", name=config.remote_name, split="train", streaming=config.stream_data)
  train_loader = data_loader.DataLoader(
      config['batch_size'], config['max_target_length'], data_root=config['root_dir'], split="train")
  eval_loader = data_loader.DataLoader(
      config['batch_size'], config['max_target_length'], data_root=config['root_dir'], split="val")