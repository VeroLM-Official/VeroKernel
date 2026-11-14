"""Host-side (NumPy in / NumPy out) helpers and CPU reference kernels. Backed by the Rust extension when it is installed, NumPy otherwise.
None of these functions import jax."""
import numpy as np

try:
    from . import _native as _impl
    NATIVE = True
except ImportError:                                                  # source checkout without the compiled extension
    from . import _fallback as _impl
    NATIVE = False

PAD_DOC = 1 << 28


def _c(x, dt): return np.ascontiguousarray(x, dtype=dt).reshape(-1)


def block_doc_ranges(doc, valid, block=128):
    """(B,T) doc ids and validity -> (lo, hi) int32 arrays of shape (B, T//block): smallest / largest document id among the VALID rows of each block
    (empty block: lo=2**30, hi=-1). These are the tables the flash-attention kernel uses to skip (q-block, k-block) pairs of unrelated documents."""
    doc = np.asarray(doc); B, T = doc.shape
    if T % block: raise ValueError(f'T={T} must be a multiple of block={block}')
    lo, hi = _impl.block_doc_ranges(_c(doc, np.int32), _c(valid, np.uint8), B, T, block)
    return np.asarray(lo).reshape(B, T // block), np.asarray(hi).reshape(B, T // block)


def valid_length(doc, pad=PAD_DOC):
    """(B,T) doc ids -> (B,) int32: index of the last real (id < pad) row + 1."""
    doc = np.asarray(doc); B, T = doc.shape
    return np.asarray(_impl.valid_length(_c(doc, np.int32), B, T, int(pad)))


def doc_ids_from_lengths(lengths, T, pad=PAD_DOC):
    """(B,n) document lengths (entries <= 0 are skipped) -> (B,T) int32 document ids; rows after the last document get `pad`."""
    lengths = np.asarray(lengths); B, n = lengths.shape
    return np.asarray(_impl.doc_ids_from_lengths(_c(lengths, np.int32), B, n, int(T), int(pad))).reshape(B, T)


def doc_ids_from_starts(starts):
    """(B,T) boolean 'a new document starts here' mask -> (B,T) int32 document ids (the first row always belongs to document 0)."""
    starts = np.asarray(starts); B, T = starts.shape
    return np.asarray(_impl.doc_ids_from_starts(_c(starts, np.uint8), B, T)).reshape(B, T)


def rwkv7_cpu(r, lw, k, v, kk, a, doc=None):
    """float32 CPU forward of the RWKV-7 recurrence, (B,T,H,N) tensors. Multi-threaded with the native extension (VEROKERNEL_THREADS=n to limit).
    Padding rows (doc >= PAD_DOC) return zeros. Meant for verification, small models and CPU inference of short sequences, not for training."""
    r = np.asarray(r); B, T, H, N = r.shape
    doc = np.zeros((B, T), np.int32) if doc is None else np.asarray(doc)
    y = _impl.wkv7_forward(*[_c(x, np.float32) for x in (r, lw, k, v, kk, a)], _c(doc, np.int32), B, T, H, N)
    return np.asarray(y).reshape(B, T, H, N)


def attention_cpu(q, k, v, doc=None, valid=None, causal=True):
    """float32 CPU forward of the document-masked attention, (B,H,T,hd) tensors. Returns (o, max_logit)."""
    q = np.asarray(q); B, H, T, hd = q.shape
    doc = np.zeros((B, T), np.int32) if doc is None else np.asarray(doc)
    valid = np.ones((B, T), bool) if valid is None else np.asarray(valid)
    o, mx = _impl.attention_forward(_c(q, np.float32), _c(k, np.float32), _c(v, np.float32), _c(doc, np.int32), _c(valid, np.uint8), B, H, T, hd, bool(causal))
    return np.asarray(o).reshape(B, H, T, hd), float(mx)


def num_threads(): return _impl.num_threads()
