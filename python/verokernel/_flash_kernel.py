"""Flash attention for TPU (Pallas): forward + dq / dkv backward, document- and validity-masked, causal or bidirectional.

The score matrix is never materialised. Block skipping: for every (batch, block) the smallest / largest document id among its VALID rows is passed as a
scalar-prefetch array; a (q-block, k-block) pair is computed only if the document ranges overlap (and, in causal mode, the k block is not entirely in the
future). Blocks made only of padding rows and blocks of unrelated documents therefore cost nothing. Skipped q blocks produce exact zeros.
"""
import functools, math
import jax, jax.numpy as jnp
from jax import lax
from ._common import (pl, pltpu, _bs, _call, NEG, _shard_map, _data_parallel_ok)

BIG_I = 1 << 30

def _fdiv(x, d):
    """floor division of a NON-NEGATIVE traced int32 by a static int via lax.div (jnp's // adds a sign() correction that newer Mosaic lowerings only support with a TPU target)"""
    return lax.div(x, jnp.int32(d))

def _mask_block(qd, kd, qi, ki, bq, bk, causal):
    m = qd == kd                                                                            # (bq,1) == (1,bk)
    if causal:
        row = qi * bq + lax.broadcasted_iota(jnp.int32, (bq, bk), 0); col = ki * bk + lax.broadcasted_iota(jnp.int32, (bq, bk), 1)
        m = m & (row >= col)
    return m

def _pair_live(qlo, qhi, klo, khi, b, qi, ki, nQ, nK, bq, bk, causal):
    live = (qhi[b * nQ + qi] >= klo[b * nK + ki]) & (qlo[b * nQ + qi] <= khi[b * nK + ki])
    return (live & (ki * bk <= qi * bq + bq - 1)) if causal else live

_HALF = -0.5e30
_NT = (((1,), (1,)), ((), ()))
_TN = (((0,), (0,)), ((), ()))

def _cases(live, qi, ki, bq, bk, causal, run):
    """run(need_causal_mask): the causal row>=col compare is only built for pairs that cross the diagonal (interior pairs need the document mask only)"""
    if causal:
        diag = (ki * bk + bk - 1) > (qi * bq)
        @pl.when(live & diag)
        def _():
            run(True)
        @pl.when(live & jnp.logical_not(diag))
        def _():
            run(False)
    else:
        @pl.when(live)
        def _():
            run(False)

# The 1/sqrt(hd) scale is folded into q once (outside the kernels), so none of the three kernels multiplies a (bq,bk) tile or the accumulators by it.
def _attn_fwd_kernel(qlo, qhi, klo, khi, q_ref, k_ref, v_ref, qd_ref, kd_ref, o_ref, lse_ref, mx_ref, m_s, l_s, acc_s, *, H, bq, bk, nQ, nK, causal):
    bh = pl.program_id(0); qi = pl.program_id(1); ki = pl.program_id(2); b = _fdiv(bh, H)
    @pl.when(ki == 0)
    def _():
        m_s[...] = jnp.full_like(m_s, NEG); l_s[...] = jnp.zeros_like(l_s); acc_s[...] = jnp.zeros_like(acc_s)
    def run(need):
        q = q_ref[0]; k = k_ref[0]; v = v_ref[0]
        s = lax.dot_general(q, k, _NT, preferred_element_type=jnp.float32)
        s = jnp.where(_mask_block(qd_ref[0], kd_ref[0], qi, ki, bq, bk, need), s, NEG)
        m_prev = m_s[...]; m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
        p = jnp.exp(s - jnp.maximum(m_new, _HALF))                                          # rows without a valid key so far: exp(NEG - HALF) = 0, no select on the (bq,bk) tile needed
        alpha = jnp.exp(m_prev - m_new)
        l_s[...] = alpha * l_s[...] + jnp.sum(p, axis=1, keepdims=True)
        acc_s[...] = alpha * acc_s[...] + jnp.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32)
        m_s[...] = m_new
    _cases(_pair_live(qlo, qhi, klo, khi, b, qi, ki, nQ, nK, bq, bk, causal), qi, ki, bq, bk, causal, run)
    @pl.when(ki == nK - 1)
    def _():
        l = jnp.maximum(l_s[...], 1e-30)                                                    # rows without any valid key (padding queries / skipped blocks) output 0
        o_ref[0] = (acc_s[...] / l).astype(o_ref.dtype)
        lse_ref[0] = m_s[...] + jnp.log(l)
        mx_ref[0, 0] = jnp.maximum(jnp.max(m_s[...], keepdims=True), -1e4)                  # max logit of the block = max of the running row maxima (no extra masked reduce per pair)

def _attn_dq_kernel(qlo, qhi, klo, khi, q_ref, k_ref, v_ref, do_ref, lse_ref, dl_ref, qd_ref, kd_ref, dq_ref, acc_s, *, H, bq, bk, nQ, nK, causal):
    bh = pl.program_id(0); qi = pl.program_id(1); ki = pl.program_id(2); b = _fdiv(bh, H)
    @pl.when(ki == 0)
    def _():
        acc_s[...] = jnp.zeros_like(acc_s)
    def run(need):
        q = q_ref[0]; k = k_ref[0]; v = v_ref[0]; do = do_ref[0]
        s = lax.dot_general(q, k, _NT, preferred_element_type=jnp.float32)
        p = jnp.where(_mask_block(qd_ref[0], kd_ref[0], qi, ki, bq, bk, need), jnp.exp(s - lse_ref[0]), 0.0)
        ds = p * (lax.dot_general(do, v, _NT, preferred_element_type=jnp.float32) - dl_ref[0])
        acc_s[...] += jnp.dot(ds.astype(k.dtype), k, preferred_element_type=jnp.float32)
    _cases(_pair_live(qlo, qhi, klo, khi, b, qi, ki, nQ, nK, bq, bk, causal), qi, ki, bq, bk, causal, run)
    @pl.when(ki == nK - 1)
    def _():
        dq_ref[0] = acc_s[...].astype(dq_ref.dtype)

def _attn_dkv_kernel(qlo, qhi, klo, khi, q_ref, k_ref, v_ref, do_ref, lse_ref, dl_ref, qd_ref, kd_ref, dk_ref, dv_ref, dk_s, dv_s, *, H, bq, bk, nQ, nK, causal):
    bh = pl.program_id(0); ki = pl.program_id(1); qi = pl.program_id(2); b = _fdiv(bh, H)
    @pl.when(qi == 0)
    def _():
        dk_s[...] = jnp.zeros_like(dk_s); dv_s[...] = jnp.zeros_like(dv_s)
    def run(need):
        q = q_ref[0]; k = k_ref[0]; v = v_ref[0]; do = do_ref[0]
        s = lax.dot_general(q, k, _NT, preferred_element_type=jnp.float32)
        p = jnp.where(_mask_block(qd_ref[0], kd_ref[0], qi, ki, bq, bk, need), jnp.exp(s - lse_ref[0]), 0.0)
        ds = p * (lax.dot_general(do, v, _NT, preferred_element_type=jnp.float32) - dl_ref[0])
        dv_s[...] += lax.dot_general(p.astype(do.dtype), do, _TN, preferred_element_type=jnp.float32)
        dk_s[...] += lax.dot_general(ds.astype(q.dtype), q, _TN, preferred_element_type=jnp.float32)
    _cases(_pair_live(qlo, qhi, klo, khi, b, qi, ki, nQ, nK, bq, bk, causal), qi, ki, bq, bk, causal, run)
    @pl.when(qi == nQ - 1)
    def _():
        dk_ref[0] = dk_s[...].astype(dk_ref.dtype); dv_ref[0] = dv_s[...].astype(dv_ref.dtype)


@functools.lru_cache(maxsize=None)
def _make_flash_core(H, bq, bk, causal):
    def last_k(qi): return _fdiv(qi * bq + bq - 1, bk)
    def fwd_impl(q, k, v, qd, kd, qlo, qhi, klo, khi):
        BH, T, hd = q.shape; nQ, nK = T // bq, T // bk
        def kidx(i, qi, ki, *_): return (i, jnp.minimum(ki, last_k(qi)) if causal else ki, 0)
        def kdidx(i, qi, ki, *_): return (_fdiv(i, H), 0, jnp.minimum(ki, last_k(qi)) if causal else ki)
        qrow = lambda i, qi, ki, *_: (i, qi, 0)
        o, lse, mx = _call(functools.partial(_attn_fwd_kernel, H=H, bq=bq, bk=bk, nQ=nQ, nK=nK, causal=causal), (BH, nQ, nK),
            [_bs((1, bq, hd), qrow), _bs((1, bk, hd), kidx), _bs((1, bk, hd), kidx), _bs((1, bq, 1), lambda i, qi, ki, *_: (_fdiv(i, H), qi, 0)), _bs((1, 1, bk), kdidx)],
            [_bs((1, bq, hd), qrow), _bs((1, bq, 1), qrow), _bs((1, 1, 1, 1), lambda i, qi, ki, *_: (i, qi, 0, 0))],
            [jax.ShapeDtypeStruct((BH, T, hd), q.dtype), jax.ShapeDtypeStruct((BH, T, 1), jnp.float32), jax.ShapeDtypeStruct((BH, nQ, 1, 1), jnp.float32)],
            [pltpu.VMEM((bq, 1), jnp.float32), pltpu.VMEM((bq, 1), jnp.float32), pltpu.VMEM((bq, hd), jnp.float32)],
            ('parallel', 'parallel', 'arbitrary'), nsp=4)(qlo, qhi, klo, khi, q, k, v, qd, kd)
        return o, lse, jnp.max(mx)
    @jax.custom_vjp
    def core(q, k, v, qd, kd, qlo, qhi, klo, khi):
        o, _, mx = fwd_impl(q, k, v, qd, kd, qlo, qhi, klo, khi); return o, mx
    def core_fwd(q, k, v, qd, kd, qlo, qhi, klo, khi):
        o, lse, mx = fwd_impl(q, k, v, qd, kd, qlo, qhi, klo, khi); return (o, mx), (q, k, v, qd, kd, qlo, qhi, klo, khi, o, lse)
    def core_bwd(res, g):
        q, k, v, qd, kd, qlo, qhi, klo, khi, o, lse = res; do = g[0]
        BH, T, hd = q.shape; nQ, nK = T // bq, T // bk
        delta = jnp.sum(do.astype(jnp.float32) * o.astype(jnp.float32), axis=-1, keepdims=True)
        def kidx(i, qi, ki, *_): return (i, jnp.minimum(ki, last_k(qi)) if causal else ki, 0)
        def kdidx(i, qi, ki, *_): return (_fdiv(i, H), 0, jnp.minimum(ki, last_k(qi)) if causal else ki)
        qrow = lambda i, qi, ki, *_: (i, qi, 0)
        kw = dict(H=H, bq=bq, bk=bk, nQ=nQ, nK=nK, causal=causal)
        dq = _call(functools.partial(_attn_dq_kernel, **kw), (BH, nQ, nK),
                   [_bs((1, bq, hd), qrow), _bs((1, bk, hd), kidx), _bs((1, bk, hd), kidx), _bs((1, bq, hd), qrow), _bs((1, bq, 1), qrow), _bs((1, bq, 1), qrow),
                    _bs((1, bq, 1), lambda i, qi, ki, *_: (_fdiv(i, H), qi, 0)), _bs((1, 1, bk), kdidx)],
                   _bs((1, bq, hd), qrow), jax.ShapeDtypeStruct((BH, T, hd), q.dtype), [pltpu.VMEM((bq, hd), jnp.float32)],
                   ('parallel', 'parallel', 'arbitrary'), nsp=4)(qlo, qhi, klo, khi, q, k, v, do, lse, delta, qd, kd)
        def first_q(ki): return _fdiv(ki * bk, bq)
        def qix(i, ki, qi, *_): return (i, jnp.maximum(qi, first_q(ki)) if causal else qi, 0)
        def qdix(i, ki, qi, *_): return (_fdiv(i, H), jnp.maximum(qi, first_q(ki)) if causal else qi, 0)
        krow = lambda i, ki, qi, *_: (i, ki, 0)
        dk, dv = _call(functools.partial(_attn_dkv_kernel, **kw), (BH, nK, nQ),
                       [_bs((1, bq, hd), qix), _bs((1, bk, hd), krow), _bs((1, bk, hd), krow), _bs((1, bq, hd), qix), _bs((1, bq, 1), qix), _bs((1, bq, 1), qix),
                        _bs((1, bq, 1), qdix), _bs((1, 1, bk), lambda i, ki, qi, *_: (_fdiv(i, H), 0, ki))],
                       [_bs((1, bk, hd), krow), _bs((1, bk, hd), krow)],
                       [jax.ShapeDtypeStruct((BH, T, hd), k.dtype), jax.ShapeDtypeStruct((BH, T, hd), v.dtype)],
                       [pltpu.VMEM((bk, hd), jnp.float32), pltpu.VMEM((bk, hd), jnp.float32)], ('parallel', 'parallel', 'arbitrary'), nsp=4)(qlo, qhi, klo, khi, q, k, v, do, lse, delta, qd, kd)
        return dq, dk, dv, None, None, None, None, None, None
    core.defvjp(core_fwd, core_bwd)
    return core

def _block_ranges(doc, valid, blk):
    """(B,T) -> flat (B*T/blk,) smallest / largest document id among the valid rows of every block (empty block: lo=BIG_I, hi=-1)"""
    B, T = doc.shape; n = T // blk
    d = doc.reshape(B, n, blk); v = valid.reshape(B, n, blk)
    return jnp.min(jnp.where(v, d, BIG_I), -1).reshape(-1).astype(jnp.int32), jnp.max(jnp.where(v, d, -1), -1).reshape(-1).astype(jnp.int32)

def make_attn(block=128):
    """returns fn(q, k, v, pdoc, valid, masked) -> (o, max_logit). q,k,v (B,H,T,hd) (already qk-normalised), pdoc (B,T) int, valid (B,T) bool.
    A query attends to keys of its own document that are valid (and, unless masked=True, not later than itself). Outputs of padding queries are 0."""
    def local(q, k, v, pdoc, valid, masked):
        B, H, T, hd = q.shape; Tp = -(-T // block) * block
        qd = pdoc.astype(jnp.int32); kd = jnp.where(valid, pdoc, -2).astype(jnp.int32); vq = valid
        q = (q.astype(jnp.float32) * (1.0 / math.sqrt(hd))).astype(q.dtype)              # softmax scale folded into q once (the autodiff of this multiply supplies the scale of dq)
        if Tp != T:
            pad = ((0, 0), (0, 0), (0, Tp - T), (0, 0)); q, k, v = (jnp.pad(t, pad) for t in (q, k, v))
            qd = jnp.pad(qd, ((0, 0), (0, Tp - T)), constant_values=-3); kd = jnp.pad(kd, ((0, 0), (0, Tp - T)), constant_values=-2); vq = jnp.pad(vq, ((0, 0), (0, Tp - T)))
        qlo, qhi = _block_ranges(qd, vq, block); klo, khi = _block_ranges(kd, vq, block)
        core = _make_flash_core(H, block, block, not masked)
        o, mx = core(q.reshape(B * H, Tp, hd), k.reshape(B * H, Tp, hd), v.reshape(B * H, Tp, hd), qd[..., None], kd[:, None, :], qlo, qhi, klo, khi)
        return o.reshape(B, H, Tp, hd)[:, :, :T], mx
    def fn(q, k, v, pdoc, valid, masked):
        if _data_parallel_ok(q.shape[0]):
            from jax.sharding import PartitionSpec as P
            sp = P('data')
            def inner(q, k, v, pdoc, valid):
                o, mx = local(q, k, v, pdoc, valid, masked); return o, lax.pmax(lax.stop_gradient(mx), 'data')
            return _shard_map(inner, (sp,) * 5, (sp, P()))(q, k, v, pdoc, valid)
        return local(q, k, v, pdoc, valid, masked)
    return fn


