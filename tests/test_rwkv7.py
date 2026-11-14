"""RWKV-7 kernel: forward and gradients against the token-by-token recurrence / float32-highest chunked reference. Pallas runs in interpret mode on CPU."""
import numpy as np
import pytest
import jax, jax.numpy as jnp
import verokernel as vk
from verokernel import reference as ref
from verokernel.autotune import rwkv7_inputs

PAD = vk.PAD_DOC


def rel(a, b, m=None):
    if m is not None: a, b = jnp.where(m, a, 0), jnp.where(m, b, 0)
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-30))


def _mask(doc): return (doc < PAD)[:, :, None, None]


@pytest.mark.parametrize('B,T,H,N', [(1, 128, 2, 64), (2, 256, 4, 32), (1, 128, 1, 128)])
def test_pallas_forward_matches_sequential(B, T, H, N):
    xs, doc, _ = rwkv7_inputs(B, T, H, N)
    y = vk.rwkv7(*xs, doc=doc, backend='pallas')
    with jax.default_matmul_precision('highest'): ys = ref.wkv7_sequential(*xs, doc=doc)
    assert rel(y, ys, _mask(doc)) < 5e-5
    assert float(jnp.abs(jnp.where(_mask(doc), 0, y)).max()) == 0.0              # padding rows are exact zeros


def test_length_not_multiple_of_16_or_128():
    xs, doc, _ = rwkv7_inputs(1, 128, 2, 64)
    xs = [x[:, :100] for x in xs]; doc = doc[:, :100]
    y = vk.rwkv7(*xs, doc=doc, backend='pallas'); assert y.shape == (1, 100, 2, 64)
    with jax.default_matmul_precision('highest'): ys = ref.wkv7_sequential(*xs, doc=doc)
    assert rel(y, ys, _mask(doc)) < 5e-5


def test_no_doc_ids_means_one_document():
    xs, _, _ = rwkv7_inputs(1, 128, 2, 64)
    y = vk.rwkv7(*xs, backend='pallas')
    with jax.default_matmul_precision('highest'): ys = ref.wkv7_sequential(*xs)
    assert rel(y, ys) < 5e-5


def test_xla_backend_matches_pallas():
    xs, doc, _ = rwkv7_inputs(1, 128, 2, 64)
    assert rel(vk.rwkv7(*xs, doc=doc, backend='xla'), vk.rwkv7(*xs, doc=doc, backend='pallas'), _mask(doc)) < 5e-5


def test_rust_cpu_matches_jax():
    xs, doc, _ = rwkv7_inputs(1, 128, 2, 64)
    yc = vk.rwkv7_cpu(*[np.asarray(x) for x in xs], np.asarray(doc))
    with jax.default_matmul_precision('highest'): ys = ref.wkv7_sequential(*xs, doc=doc)
    assert rel(jnp.asarray(yc), ys, _mask(doc)) < 5e-6


@pytest.mark.parametrize('corr', [0.0, 0.98, 1.0])
def test_gradients_match_reference(corr):
    """corr=1.0: all keys of a row are identical (repeated text): the case that makes long-chunk triangular solves explode."""
    B, T, H, N = 1, 128, 2, 64
    xs, doc, w = rwkv7_inputs(B, T, H, N, corr)
    f = lambda *x: jnp.sum(vk.rwkv7(*x, doc=doc, backend='pallas') * w)
    g = jax.grad(f, argnums=tuple(range(6)))(*xs)
    with jax.default_matmul_precision('highest'):
        rg = jax.grad(lambda *x: jnp.sum(ref.wkv7_sequential(*x, doc=doc) * w), argnums=tuple(range(6)))(*xs)
    for name, a, b in zip('r lw k v kk a'.split(), g, rg):
        assert bool(jnp.all(jnp.isfinite(a))), name
        assert rel(a, b) < 2e-3, (name, rel(a, b))


def test_support_check():
    assert vk.rwkv7_supported((1, 128, 2, 64))[0] and vk.rwkv7_supported((1, 128, 1, 128))[0]
    assert not vk.rwkv7_supported((1, 128, 3, 64))[0]            # odd number of heads: a 128-lane block holds two 64-wide heads
    assert not vk.rwkv7_supported((1, 128, 2, 48))[0]
    with pytest.raises(ValueError): vk.rwkv7(*rwkv7_inputs(1, 128, 3, 64)[0], backend='pallas')


def test_auto_backend_falls_back_on_unsupported_shape():
    xs, doc, _ = rwkv7_inputs(1, 128, 3, 64)
    y = vk.rwkv7(*xs, doc=doc)                                    # auto -> xla on CPU
    with jax.default_matmul_precision('highest'): ys = ref.wkv7_sequential(*xs, doc=doc)
    assert rel(y, ys, _mask(doc)) < 5e-5
