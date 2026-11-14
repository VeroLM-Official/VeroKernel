"""Time rwkv7 / attention (forward+backward) on the current device, Pallas vs XLA fallback.
usage: python benchmarks/bench.py [--B 8 --T 4096 --H 16 --N 64 --iters 10]"""
import argparse, time
import jax, jax.numpy as jnp
import verokernel as vk
from verokernel.autotune import rwkv7_inputs

p = argparse.ArgumentParser()
for n, d in (('B', 8), ('T', 4096), ('H', 16), ('N', 64), ('iters', 10)): p.add_argument('--' + n, type=int, default=d)
a = p.parse_args()


def timeit(f, args):
    jax.block_until_ready(f(*args)); t0 = time.perf_counter()
    for _ in range(a.iters): out = f(*args)
    jax.block_until_ready(out); return (time.perf_counter() - t0) / a.iters * 1e3


print(vk.info())
xs, doc, w = rwkv7_inputs(a.B, a.T, a.H, a.N)
for be in ('xla', 'pallas'):
    if be == 'pallas' and not vk.rwkv7_supported((a.B, a.T, a.H, a.N))[0]: continue
    f = jax.jit(jax.value_and_grad(lambda *x, be=be: jnp.sum(vk.rwkv7(*x, doc=doc, backend=be) * w), argnums=tuple(range(6))))
    print(f'rwkv7     {be:7s} fwd+bwd {timeit(f, xs):9.2f} ms')
hd = a.N; k = jax.random.split(jax.random.PRNGKey(0), 3)
q, kx, v = (jax.random.normal(k[i], (a.B, a.H, a.T, hd), dtype=jnp.bfloat16) for i in range(3))
for be in ('xla', 'pallas'):
    f = jax.jit(jax.value_and_grad(lambda q, k_, v_, be=be: jnp.sum(vk.attention(q, k_, v_, doc, doc < vk.PAD_DOC, backend=be).astype(jnp.float32)), argnums=(0, 1, 2)))
    print(f'attention {be:7s} fwd+bwd {timeit(f, (q, kx, v)):9.2f} ms')
