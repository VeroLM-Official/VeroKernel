"""Rust extension vs the NumPy fallback and basic semantics (no jax needed)."""
import numpy as np
import pytest
import verokernel as vk
from verokernel import _fallback as fb

rng = np.random.default_rng(0)
sg = lambda x: 1 / (1 + np.exp(-x))


def _rwkv_inputs(B=2, T=48, H=3, N=16):
    r, k, v = [rng.normal(size=(B, T, H, N)).astype(np.float32) for _ in range(3)]
    lw = (-sg(rng.normal(size=(B, T, H, N))) * 0.6).astype(np.float32)
    kk = rng.normal(size=(B, T, H, N)).astype(np.float32); kk /= np.linalg.norm(kk, axis=-1, keepdims=True)
    a = sg(rng.normal(size=(B, T, H, N))).astype(np.float32)
    doc = np.zeros((B, T), np.int32); doc[:, 20:] = 1; doc[:, 40:] = vk.PAD_DOC
    return (r, lw, k, v, kk, a), doc


def test_native_extension_loaded():
    assert vk.NATIVE, 'the compiled Rust extension is not importable'


def test_rwkv7_rust_matches_numpy():
    xs, doc = _rwkv_inputs(); B, T, H, N = xs[0].shape
    y = vk.rwkv7_cpu(*xs, doc)
    y2 = fb.wkv7_forward(*[x.ravel() for x in xs], doc.ravel(), B, T, H, N).reshape(B, T, H, N)
    np.testing.assert_allclose(y, y2, atol=2e-5, rtol=1e-5)
    assert np.all(y[:, 40:] == 0)


def test_rwkv7_document_isolation():
    xs, doc = _rwkv_inputs()
    y = vk.rwkv7_cpu(*xs, doc)
    cut = [x[:, 20:40] for x in xs]                                        # the second document alone
    y_alone = vk.rwkv7_cpu(*cut, np.zeros((2, 20), np.int32))
    np.testing.assert_allclose(y[:, 20:40], y_alone, atol=1e-5)


@pytest.mark.parametrize('causal', [True, False])
def test_attention_rust_matches_numpy(causal):
    B, H, T, hd = 2, 3, 40, 16
    q, k, v = [rng.normal(size=(B, H, T, hd)).astype(np.float32) for _ in range(3)]
    doc = np.zeros((B, T), np.int32); doc[:, 15:] = 1; doc[:, 33:] = 9; valid = np.ones((B, T), bool); valid[:, 33:] = False; valid[0, 3] = False
    o, mx = vk.attention_cpu(q, k, v, doc, valid, causal)
    o2, mx2 = fb.attention_forward(q.ravel(), k.ravel(), v.ravel(), doc.ravel(), valid.ravel(), B, H, T, hd, causal)
    np.testing.assert_allclose(o, o2.reshape(o.shape), atol=2e-6)
    assert abs(mx - mx2) < 1e-5
    assert np.all(o[:, :, 33:] == 0)


def test_host_helpers_match_fallback():
    B, T = 3, 64
    doc = np.sort(rng.integers(0, 5, size=(B, T)), axis=1).astype(np.int32); valid = rng.random((B, T)) > 0.3
    lo, hi = vk.block_doc_ranges(doc, valid, 8)
    lo2, hi2 = fb.block_doc_ranges(doc.ravel(), valid.ravel(), B, T, 8)
    assert (lo.ravel() == lo2).all() and (hi.ravel() == hi2).all()
    assert (vk.valid_length(doc) == fb.valid_length(doc.ravel(), B, T, vk.PAD_DOC)).all()
    lengths = rng.integers(-1, 20, size=(B, 6)).astype(np.int32)
    assert (vk.doc_ids_from_lengths(lengths, T) == fb.doc_ids_from_lengths(lengths.ravel(), B, 6, T, vk.PAD_DOC).reshape(B, T)).all()
    starts = rng.random((B, T)) > 0.8
    assert (vk.doc_ids_from_starts(starts) == fb.doc_ids_from_starts(starts.ravel(), B, T).reshape(B, T)).all()


def test_doc_ids_examples():
    ids = vk.doc_ids_from_lengths([[3, 0, 2]], 8)
    assert ids.tolist() == [[0, 0, 0, 1, 1, vk.PAD_DOC, vk.PAD_DOC, vk.PAD_DOC]]
    assert vk.doc_ids_from_starts(np.array([[1, 0, 1, 0, 1]])).tolist() == [[0, 0, 1, 1, 2]]


def test_shape_errors_are_python_exceptions():
    with pytest.raises(ValueError):
        vk.block_doc_ranges(np.zeros((1, 10), np.int32), np.ones((1, 10), bool), 4)


def test_import_does_not_load_jax():
    import subprocess, sys, os
    code = "import sys, verokernel; assert 'jax' not in sys.modules; print('ok')"
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(vk.__file__)))
    assert subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True).stdout.strip() == 'ok'
