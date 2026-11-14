"""Public JAX API: rwkv7(...) and attention(...). Pallas kernels on TPU, exact XLA implementations everywhere else."""
import functools
import jax, jax.numpy as jnp
from . import reference as _ref
from ._common import PAD_DOC, set_mesh, get_mesh

CHUNK = 16            # solve-block length of the RWKV-7 kernel (fixed: longer blocks are numerically unsafe, see the README)
_BACKENDS = ('auto', 'pallas', 'xla')


def rwkv7_supported(shape):
    """(ok, reason) for the Pallas RWKV-7 kernel. shape = (B, T, H, N). T is padded to a multiple of 16 automatically, so only H and N matter."""
    _, _, H, N = shape
    if N >= 128:
        hp = 1
        if N % 128: return False, f'head size N={N} must be a multiple of 128 (or a divisor of 128)'
    else:
        if 128 % N: return False, f'head size N={N} must divide 128 (or be a multiple of 128)'
        hp = 128 // N
    if H % hp: return False, f'number of heads H={H} must be a multiple of {hp} for N={N} (a 128-lane block holds {hp} heads)'
    return True, ''


def _pick(backend, ok_fn):
    if backend not in _BACKENDS: raise ValueError(f'backend must be one of {_BACKENDS}')
    if backend == 'auto':
        if jax.default_backend() != 'tpu': return 'xla'
        ok, _ = ok_fn(); return 'pallas' if ok else 'xla'
    if backend == 'pallas':
        ok, why = ok_fn()
        if not ok: raise ValueError(f'Pallas kernel not applicable: {why}')
    return backend


@functools.lru_cache(maxsize=None)
def _rwkv7_fn(precision, vmem_mb):
    from ._rwkv7_kernel import make_rwkv7
    return make_rwkv7(precision, vmem_mb=vmem_mb)


def rwkv7(r, lw, k, v, kk, a, doc=None, *, backend='auto', precision='mixed', vmem_mb=None):
    """RWKV-7 (generalised delta rule) token mixing core, differentiable (fused forward and backward kernels).

    r, lw, k, v, kk, a : (B, T, H, N) float tensors (cast to float32). lw is the LOG decay (<= 0), kk the L2-normalised removal key, a the in-context learning rate.
        The recurrence per head (state S is N x N):  S <- S * exp(lw)[None, :] - (S @ kk)[:, None] * (kk * a)[None, :] + v[:, None] * k[None, :];   y = S @ r
    doc : optional (B, T) int32 document ids. The state is reset where the id changes (documents packed into one row never see each other).
        Rows with id >= PAD_DOC (verokernel.PAD_DOC) are padding: they are skipped and return exact zeros. Ids must be non-decreasing to be efficient but any ids work.
    backend : 'auto' (Pallas on TPU when the shape is supported, XLA otherwise) | 'pallas' | 'xla'
    precision : 'mixed' (default, fastest accurate setting) | 'high' (all dots in 3-pass bf16, slower)
    Returns y (B, T, H, N) float32.
    """
    B, T, H, N = r.shape
    f32 = lambda x: x.astype(jnp.float32)
    r, lw, k, v, kk, a = map(f32, (r, lw, k, v, kk, a))
    doc = jnp.zeros((B, T), jnp.int32) if doc is None else doc.astype(jnp.int32)
    Tp = -(-T // CHUNK) * CHUNK
    if Tp != T:
        pad4 = ((0, 0), (0, Tp - T), (0, 0), (0, 0))
        r, lw, k, v, kk, a = (jnp.pad(x, pad4) for x in (r, lw, k, v, kk, a)); doc = jnp.pad(doc, ((0, 0), (0, Tp - T)), constant_values=PAD_DOC)
    be = _pick(backend, lambda: rwkv7_supported((B, Tp, H, N)))
    if be == 'pallas':
        y = _rwkv7_fn(precision, vmem_mb)(r, lw, k, v, kk, a, CHUNK, doc)
    else:
        y = _ref.wkv7_chunked(r, lw, k, v, kk, a, CHUNK, doc=doc)
    return (y * (doc < PAD_DOC)[:, :, None, None])[:, :T]                  # padding rows are exact zeros on every backend


def attention_supported(shape, block=128):
    """(ok, reason) for the Pallas attention kernel. shape = (B, H, T, hd)."""
    hd = shape[-1]
    if hd % 8: return False, f'head size {hd} must be a multiple of 8'
    if block % 128: return False, f'block={block} must be a multiple of 128'
    return True, ''


@functools.lru_cache(maxsize=None)
def _attn_fn(block):
    from ._flash_kernel import make_attn
    return make_attn(block)


def attention(q, k, v, doc=None, valid=None, *, causal=True, block=128, backend='auto', return_max_logit=False):
    """Flash attention with document and validity masking, differentiable (fused forward / dq / dkv kernels, no (T,T) score matrix).

    q, k, v : (B, H, T, hd). Scores are q.k / sqrt(hd) (apply qk-norm / RoPE before calling). bf16 or float32.
    doc   : optional (B, T) int ids. A query only attends to keys of its own document.   valid : optional (B, T) bool. Keys with valid=False are never attended to.
    causal: True = key position <= query position; False = bidirectional inside the document.
    Queries with no attendable key (padding) return zeros. T need not be a multiple of `block` (it is padded internally).
    return_max_logit: also return the largest attention logit (scalar), e.g. for QK-clip / logit monitoring.
    """
    B, H, T, hd = q.shape
    doc = jnp.zeros((B, T), jnp.int32) if doc is None else doc.astype(jnp.int32)
    valid = jnp.ones((B, T), bool) if valid is None else valid.astype(bool)
    be = _pick(backend, lambda: attention_supported(q.shape, block))
    if be == 'pallas':
        o, mx = _attn_fn(block)(q, k, v, doc, valid, not causal)
    else:
        o, mx = _ref.attention_xla(q, k, v, doc, valid, causal, block)
    return (o, mx) if return_max_logit else o
