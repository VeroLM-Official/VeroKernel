"""Lower every kernel for TPU (Mosaic) on a CPU host with jax.export. This checks the lowering rules and shape constraints; it does not execute on a TPU."""
import pytest
import jax, jax.numpy as jnp
from jax import export
import verokernel as vk
from verokernel import _common

S = jax.ShapeDtypeStruct


@pytest.fixture(autouse=True)
def force_tpu_lowering():
    _common._FORCE_TPU_LOWERING = True
    yield
    _common._FORCE_TPU_LOWERING = False


def _lower(f, *specs):
    txt = export.export(jax.jit(f), platforms=['tpu'])(*specs).mlir_module()
    assert 'tpu_custom_call' in txt


@pytest.mark.parametrize('B,T,H,N', [(2, 512, 8, 64), (1, 256, 4, 128), (1, 256, 16, 32), (1, 200, 2, 64)])
def test_rwkv7_lowers(B, T, H, N):
    x = S((B, T, H, N), jnp.float32); d = S((B, T), jnp.int32)
    _lower(lambda *a: vk.rwkv7(*a[:6], doc=a[6], backend='pallas'), *[x] * 6, d)
    _lower(lambda *a: jax.grad(lambda *y: jnp.sum(vk.rwkv7(*y[:6], doc=y[6], backend='pallas')), argnums=tuple(range(6)))(*a), *[x] * 6, d)


@pytest.mark.parametrize('B,H,T,hd,dt', [(2, 8, 512, 64, jnp.bfloat16), (1, 4, 1000, 128, jnp.bfloat16), (1, 4, 512, 256, jnp.bfloat16), (1, 4, 512, 64, jnp.float32)])
@pytest.mark.parametrize('causal', [True, False])
def test_attention_lowers(B, H, T, hd, dt, causal):
    a = S((B, H, T, hd), dt); d = S((B, T), jnp.int32); v = S((B, T), jnp.bool_)
    _lower(lambda q, k, vv, dd, va: vk.attention(q, k, vv, dd, va, causal=causal, backend='pallas'), a, a, a, d, v)
    _lower(lambda q, k, vv, dd, va: jax.grad(lambda q, k, vv: jnp.sum(vk.attention(q, k, vv, dd, va, causal=causal, backend='pallas').astype(jnp.float32)), argnums=(0, 1, 2))(q, k, vv), a, a, a, d, v)
