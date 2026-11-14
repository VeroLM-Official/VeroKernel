import numpy as np
import pytest
import jax, jax.numpy as jnp
import verokernel as vk
from verokernel import reference as ref


def _inputs(B, H, T, hd, dtype=jnp.float32, seed=0):
    rng = np.random.default_rng(seed)
    q, k, v = (jnp.asarray(rng.normal(size=(B, H, T, hd)).astype(np.float32)) for _ in range(3))
    nrm = lambda x: x / jnp.sqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)
    q, k = nrm(q), nrm(k)
    doc = np.zeros((B, T), np.int32); valid = np.ones((B, T), bool)
    for b in range(B):
        nv = T - T // 6 - b; valid[b, nv:] = False; doc[b, nv:] = 10 ** 6
        for c in np.sort(rng.choice(np.arange(5, nv - 5), 3, replace=False)): doc[b, c:nv] += 1
    w = jnp.asarray(rng.normal(size=(B, H, T, hd)).astype(np.float32)) * jnp.asarray(valid)[:, None, :, None]
    return q.astype(dtype), k.astype(dtype), v.astype(dtype), jnp.asarray(doc), jnp.asarray(valid), w


def rel(a, b): return float(jnp.linalg.norm(a.astype(jnp.float32) - b) / (jnp.linalg.norm(b) + 1e-30))


@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('B,H,T,hd', [(2, 2, 256, 64), (1, 2, 200, 128)])
def test_forward_matches_dense(causal, B, H, T, hd):
    q, k, v, doc, valid, _ = _inputs(B, H, T, hd)
    o, mx = vk.attention(q, k, v, doc, valid, causal=causal, backend='pallas', return_max_logit=True)
    with jax.default_matmul_precision('highest'): od = ref.attention_dense(q, k, v, doc, valid, causal)
    assert o.shape == q.shape and rel(o, od) < 2e-4
    assert bool(jnp.isfinite(mx)) and float(mx) > 0


def test_padding_queries_output_zero():
    q, k, v, doc, valid, _ = _inputs(2, 2, 256, 64)
    o = vk.attention(q, k, v, doc, valid, causal=True, backend='pallas')
    assert float(jnp.abs(jnp.where(valid[:, None, :, None], 0.0, o)).max()) == 0.0


@pytest.mark.parametrize('causal', [True, False])
def test_gradients_match_dense(causal):
    q, k, v, doc, valid, w = _inputs(1, 2, 256, 64)
    f = lambda q, k, v: jnp.sum(vk.attention(q, k, v, doc, valid, causal=causal, backend='pallas') * w)
    g = jax.grad(f, argnums=(0, 1, 2))(q, k, v)
    with jax.default_matmul_precision('highest'):
        gd = jax.grad(lambda q, k, v: jnp.sum(ref.attention_dense(q, k, v, doc, valid, causal) * w), argnums=(0, 1, 2))(q, k, v)
    for n, a, b in zip('qkv', g, gd): assert rel(a, b) < 5e-4, (n, rel(a, b))


def test_bf16():
    q, k, v, doc, valid, _ = _inputs(1, 2, 256, 64, jnp.bfloat16)
    o = vk.attention(q, k, v, doc, valid, backend='pallas')
    with jax.default_matmul_precision('highest'): od = ref.attention_dense(q, k, v, doc, valid, True)
    assert o.dtype == jnp.bfloat16 and rel(o, od) < 3e-2


def test_xla_backend_matches_dense_and_rust():
    q, k, v, doc, valid, _ = _inputs(1, 2, 256, 64)
    ox = vk.attention(q, k, v, doc, valid, backend='xla')
    with jax.default_matmul_precision('highest'): od = ref.attention_dense(q, k, v, doc, valid, True)
    assert rel(ox, od) < 1e-4
    oc, mx = vk.attention_cpu(*[np.asarray(x) for x in (q, k, v, doc, valid)])
    assert rel(jnp.asarray(oc), od) < 1e-5


def test_block_tables_from_rust_match_kernel_tables():
    from verokernel._flash_kernel import _block_ranges
    q, k, v, doc, valid, _ = _inputs(2, 1, 256, 64)
    lo, hi = _block_ranges(doc, valid, 128)
    lo2, hi2 = vk.block_doc_ranges(np.asarray(doc), np.asarray(valid), 128)
    assert (np.asarray(lo) == lo2.ravel()).all() and (np.asarray(hi) == hi2.ravel()).all()


def test_defaults_and_support():
    q, k, v, *_ = _inputs(1, 2, 128, 64)
    o = vk.attention(q, k, v, backend='pallas'); od = ref.attention_dense(q, k, v, jnp.zeros((1, 128), jnp.int32), jnp.ones((1, 128), bool), True)
    assert rel(o, od) < 2e-4
    assert not vk.attention_supported((1, 2, 128, 60))[0] and not vk.attention_supported((1, 2, 128, 64), block=96)[0]
