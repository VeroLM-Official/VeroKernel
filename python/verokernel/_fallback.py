"""Pure-NumPy implementations of everything the native (Rust) extension provides. Used automatically when the compiled extension is unavailable
(for example a source checkout without a Rust toolchain). Same results, slower."""
import numpy as np

PAD_DOC = 1 << 28
BIG_I = 1 << 30


def block_doc_ranges(doc, valid, b, t, blk):
    n = t // blk
    d = doc.reshape(b, n, blk); v = valid.reshape(b, n, blk).astype(bool)
    lo = np.where(v, d, BIG_I).min(-1).reshape(-1).astype(np.int32)
    hi = np.where(v, d, -1).max(-1).reshape(-1).astype(np.int32)
    return lo, hi


def valid_length(doc, b, t, pad):
    d = doc.reshape(b, t) < pad
    return np.where(d.any(1), t - np.argmax(d[:, ::-1], axis=1), 0).astype(np.int32)


def doc_ids_from_lengths(lengths, b, n, t, pad):
    lengths = lengths.reshape(b, n); out = np.full((b, t), pad, np.int32)
    for bi in range(b):
        pos = 0; i = 0
        for ln in lengths[bi]:
            if ln <= 0: continue
            end = min(pos + int(ln), t); out[bi, pos:end] = i; pos = end; i += 1
            if pos >= t: break
    return out.reshape(-1)


def doc_ids_from_starts(starts, b, t):
    s = (starts.reshape(b, t) != 0).astype(np.int32); s[:, 0] = 0
    return np.cumsum(s, axis=1, dtype=np.int32).reshape(-1)


def wkv7_forward(r, lw, k, v, kk, a, doc, b, t, h, n):
    r, lw, k, v, kk, a = (x.reshape(b, t, h, n).astype(np.float32) for x in (r, lw, k, v, kk, a)); doc = doc.reshape(b, t)
    y = np.zeros((b, t, h, n), np.float32)
    for bi in range(b):
        S = np.zeros((h, n, n), np.float32)
        for ti in range(t):
            if ti > 0 and doc[bi, ti] != doc[bi, ti - 1]: S[:] = 0
            if doc[bi, ti] >= PAD_DOC: continue
            skk = np.einsum('hij,hj->hi', S, kk[bi, ti])
            S = S * np.exp(lw[bi, ti])[:, None, :] - skk[:, :, None] * (kk[bi, ti] * a[bi, ti])[:, None, :] + v[bi, ti][:, :, None] * k[bi, ti][:, None, :]
            y[bi, ti] = np.einsum('hij,hj->hi', S, r[bi, ti])
    return y.reshape(-1)


def attention_forward(q, k, v, doc, valid, b, h, t, hd, causal):
    q, k, v = (x.reshape(b, h, t, hd).astype(np.float32) for x in (q, k, v)); doc = doc.reshape(b, t); valid = valid.reshape(b, t).astype(bool)
    s = np.einsum('bhqd,bhkd->bhqk', q, k) / np.float32(np.sqrt(hd))
    m = (doc[:, :, None] == doc[:, None, :]) & valid[:, None, :]
    if causal: m = m & np.tril(np.ones((t, t), bool))[None]
    m = m[:, None]
    mx = float(max(np.where(m, s, -1e4).max(), -1e4))
    s = np.where(m, s, -np.inf); any_ = m.any(-1, keepdims=True)
    s = s - np.where(any_, s.max(-1, keepdims=True), 0.0)
    p = np.where(m, np.exp(s), 0.0); p = p / np.maximum(p.sum(-1, keepdims=True), 1e-30)
    return np.einsum('bhqk,bhkd->bhqd', p, v).astype(np.float32).reshape(-1), mx


def num_threads(): return 1
