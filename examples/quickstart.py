"""Minimal usage of VeroKernel. Runs on CPU (XLA fallback) or TPU (Pallas kernels) without changes."""
import jax, jax.numpy as jnp
import numpy as np
import verokernel as vk

B, T, H, N = 2, 256, 4, 64                      # RWKV-7: batch, tokens, heads, head size
key = jax.random.split(jax.random.PRNGKey(0), 6)
r, k, v = (jax.random.normal(key[i], (B, T, H, N)) for i in range(3))
lw = -jax.nn.sigmoid(jax.random.normal(key[3], (B, T, H, N))) * 0.6065          # log decay, <= 0
kk = jax.random.normal(key[4], (B, T, H, N)); kk = kk / jnp.linalg.norm(kk, axis=-1, keepdims=True)
a = jax.nn.sigmoid(jax.random.normal(key[5], (B, T, H, N)))

# documents packed into the rows (lengths 100 + 120, the rest is padding)
doc = jnp.asarray(vk.doc_ids_from_lengths([[100, 120], [256, 0]], T))
y = vk.rwkv7(r, lw, k, v, kk, a, doc=doc)                                       # (B, T, H, N) float32, differentiable
print('rwkv7  ->', y.shape, y.dtype)

# Transformer attention: (B, H, T, hd), same document ids; keys of padding rows are masked
q, kx, vx = (jax.random.normal(key[i], (B, H, T, 64), dtype=jnp.bfloat16) for i in range(3))
valid = doc < vk.PAD_DOC
o = vk.attention(q, kx, vx, doc, valid, causal=True)
print('attention ->', o.shape, o.dtype)

grads = jax.grad(lambda *x: jnp.sum(vk.rwkv7(*x, doc=doc)), argnums=tuple(range(6)))(r, lw, k, v, kk, a)
print('grads  ->', [g.shape for g in grads][:2], '...')

# CPU reference of the same recurrence (Rust, no jax involved) - handy for debugging
y_cpu = vk.rwkv7_cpu(*[np.asarray(x) for x in (r, lw, k, v, kk, a)], np.asarray(doc))
print('max |pallas/xla - rust cpu| on real rows:', float(np.abs(np.asarray(y) - y_cpu)[np.asarray(valid)].max()))
print(vk.info())
