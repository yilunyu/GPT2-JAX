from dataclasses import dataclass
import common_types
import flax.linen as nn
from flax.linen import initializers
import jax
import jax.numpy as jnp

from layers import attentions

kernel_init = initializers.lecun_normal(dtype=jnp.float32)
embed_init = initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0)

class MLP(nn.Module):
  n_embd: int
  computation_dtype: jnp.dtype = jnp.bfloat16
  weight_dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.f1 = nn.Dense(
        4 * self.n_embd,
        param_dtype=self.weight_dtype,
        dtype=self.computation_dtype,
        kernel_init=nn.with_partitioning(kernel_init, (None, None)))
    self.f2 = nn.Dense(
        self.n_embd,
        param_dtype=self.weight_dtype,
        dtype=self.computation_dtype,
        kernel_init=nn.with_partitioning(kernel_init, (None, None)))

  def __call__(self, x):
    x = self.f1(x)
    x = nn.gelu(x)
    x = self.f2(x)
    return x

class MultiHeadAttention(nn.Module):
  num_heads: int
  head_dim: int
  mesh: jax.sharding.Mesh
  computation_dtype: jnp.dtype = jnp.bfloat16
  weight_dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.query = nn.Dense(
        features=self.num_heads * self.head_dim,
        use_bias=False,
        param_dtype=self.weight_dtype,
        dtype=self.computation_dtype,
        kernel_init=nn.with_partitioning(kernel_init, (None, None))
        )
    self.key = nn.Dense(
        features=self.num_heads * self.head_dim,
        use_bias=False,
        param_dtype=self.weight_dtype,
        dtype=self.computation_dtype,
        kernel_init=nn.with_partitioning(kernel_init, (None, None))
        )
    self.value = nn.Dense(
        features=self.num_heads * self.head_dim,
        use_bias=False,
        param_dtype=self.weight_dtype,
        dtype=self.computation_dtype,
        kernel_init=nn.with_partitioning(kernel_init, (None, None)))
    self.dense = nn.Dense(
        features=self.num_heads * self.head_dim,
        param_dtype=self.weight_dtype,
        dtype=self.computation_dtype,
        kernel_init=nn.with_partitioning(kernel_init, (None, None)))

  def __call__(self, x):
    B, T, C = x.shape
    q = self.query(x)  # (B, T, num_heads*head_dim)
    k = self.key(x)  # (B, T, num_heads*head_dim)
    v = self.value(x)  # (B, T, num_heads*head_dim)
    q = q.reshape(B, T, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
    k = k.reshape(B, T, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
    v = v.reshape(B, T, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
    wei = jnp.einsum('bnth,bnsh->bnts', q, k) / jnp.sqrt(self.head_dim)  # (B, num_heads, T, T)
    mask = jnp.tril(jnp.ones((T, T)))
    wei = jnp.where(mask, wei, -jnp.inf)
    wei = nn.softmax(wei, axis=-1)  # (B, num_heads, T, T)
    out = jnp.einsum('bnts,bnsh->bnth', wei, v)  # (B, num_heads, T, head_dim)
    out = out.transpose((0, 2, 1, 3)).reshape(B, T, self.num_heads * self.head_dim)
    return self.dense(out)  # B, T, C

class DecoderBlock(nn.Module):
  config: common_types.Config
  mesh: jax.sharding.Mesh
  computation_dtype: jnp.dtype = jnp.bfloat16
  weight_dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.norm_1 = nn.RMSNorm(self.config["embed_dim"])
    self.attention = MultiHeadAttention(
        num_heads=self.config["num_kv_heads"],
        head_dim=self.config["head_dim"],
        mesh=self.mesh,
        computation_dtype=self.computation_dtype,
        weight_dtype=self.weight_dtype,
    )

    # Flash attention from maxtext:
    # self.attention = attentions.Attention(
    #     config=self.config,
    #     num_query_heads=self.config["num_query_heads"],
    #     num_kv_heads=self.config["num_kv_heads"],
    #     head_dim=self.config["head_dim"],
    #     max_target_length=self.config["max_target_length"],
    #     mesh=self.mesh,
    #     attention_kernel=self.config["attention_kernel"],
    #     dtype=self.computation_dtype,
    #     weight_dtype=self.weight_dtype,
    #     dropout_rate=self.config["dropout_rate"],
    # )

    self.norm_2 = nn.RMSNorm(self.config["embed_dim"])
    self.mlp = MLP(
        self.config["embed_dim"],
        weight_dtype=self.weight_dtype,
        computation_dtype=self.computation_dtype)

  def __call__(self, x):
    x = self.norm_1(x)
    x = x + self.attention(x)
    return x + self.mlp(self.norm_2(x))

class GPT(nn.Module):
  config: common_types.Config
  mesh: jax.sharding.Mesh
  computation_dtype: jnp.dtype = jnp.bfloat16
  weight_dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.token_embedding = nn.Embed(
        num_embeddings=self.config["vocab_size"],
        features=self.config["embed_dim"],
        param_dtype=self.weight_dtype,
        dtype=self.computation_dtype,
        embedding_init=nn.with_partitioning(embed_init, (None, None)))
    # TODO: switch to RoPE.
    self.position_encoding = nn.Embed(
        num_embeddings=self.config["max_target_length"],
        features=self.config["embed_dim"],
        param_dtype=self.weight_dtype,
        dtype=self.computation_dtype,
        embedding_init=nn.with_partitioning(embed_init, (None, None)))
    remat_policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
    RemattedDecoderBlock = nn.remat(
        DecoderBlock,
        prevent_cse=False,
        policy=remat_policy,
    )
    self.decoder_blocks = [
        RemattedDecoderBlock(
            config=self.config,
            mesh=self.mesh,
            computation_dtype=self.computation_dtype,
            weight_dtype=self.weight_dtype)
            for _ in range(self.config["num_layers"])
    ]
    self.final_norm = nn.RMSNorm()

  def __call__(self, x):
    B, T = x.shape
    x = self.token_embedding(x)
    pos = self.position_encoding(jnp.arange(T))
    x += pos
    for block in self.decoder_blocks:
      x = block(x)
    x = self.final_norm(x)
    # weight sharing scheme.
    return self.token_embedding.attend(x)
