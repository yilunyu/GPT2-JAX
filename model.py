from dataclasses import dataclass
import common_types
import flax.linen as nn
import jax
import jax.numpy as jnp

from layers import attentions

class MLP(nn.Module):
  n_embd: int
  
  def setup(self):
    self.c_fc = nn.Dense(4 * self.n_embd, param_dtype=jnp.float32, dtype=jnp.bfloat16)
    self.c_proj = nn.Dense(self.n_embd, param_dtype=jnp.float32, dtype=jnp.bfloat16)

  def __call__(self, x):
    x = self.c_fc(x)
    x = nn.gelu(x)
    x = self.c_proj(x)
    return x

class DecoderBlock(nn.Module):
  config: common_types.Config
  mesh: jax.sharding.Mesh
  computation_dtype: jnp.dtype = jnp.bfloat16
  weight_dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.norm_1 = nn.RMSNorm(self.config.embed_dim)
    self.attention = attentions.Attention(
        config=self.config,
        num_query_heads=self.config.num_query_heads,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
        max_target_length=self.config.max_target_length,
        mesh=self.mesh,
        attention_kernel=self.config.attention_kernel,
        dtype=self.computation_dtype,
        weight_dtype=self.weight_dtype,
        dropout_rate=self.config.dropout_rate,
    )
    self.norm_2 = nn.RMSNorm(self.config.embed_dim)
    self.mlp = nn.MLP(self.config.embed_dim)

  def __call__(self, x):
    x = x + self.attention(self.norm_1(x))
    return x + self.mlp(self.norm_2(x))

class GPT(nn.Module):
  config: common_types.Config

  def setup(self):
    self.token_embedding = nn.Embed(
        num_embeddings=self.config.vocab_size,
        features=self.config.hidden_dim,
        param_dtype=jnp.float32,
        dtype=jnp.bfloat16)
    # TODO: switch to RoPE.
    self.position_encoding = nn.Embed(
        num_embeddings=self.config.max_target_length,
        features=self.config.hidden_dim,
        param_dtype=jnp.float32,
        dtype=jnp.bfloat16
    )
    remat_policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
    # TODO: apply remat_policy on DecoderBlock.
    RemattedDecoderBlock = nn.remat(
        DecoderBlock,
        prevent_cse=False,
        policy=remat_policy,
    )
    self.decoder_blocks = [
        RemattedDecoderBlock(self.config.hidden_dim) for _ in range(self.config.num_layers)
    ]
    self.final_norm = nn.RMSNorm()
    self.final_dense = nn.Dense(
        features=self.config.vocab_size,
        param_dtype=jnp.float32,
        dtype=jnp.bfloat16)

  def __call__(self, x):
    B, T = x.shape
    x = self.token_embedding(x)
    pos = self.position_encoding(jnp.arange(T))
    x += pos
    for block in self.decoder_blocks:
      x = block(x)
    x = self.final_norm(x)
    # weight sharing scheme.
    kernel = self.token_embedding.variables['params']['embedding']
    return self.final_dense.apply({'params': {'kernel': kernel}}, x)
