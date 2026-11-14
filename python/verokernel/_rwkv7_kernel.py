"""RWKV-7 chunked delta-rule kernel (forward + hand-fused backward) for TPU, written in Pallas.

Numerics: the within-chunk triangular solve (I + Bc)^-1 has entries up to ~2^(L-2) when the keys of a chunk are alike (repeated text, indentation, padding), so
solve blocks longer than 16 tokens lose all digits (NaN / 1e14 errors were measured on correlated keys, even in float32). The kernel therefore keeps solve blocks
of L=16 tokens and packs G=8 of them into one 128-token program: every dense matrix op (interactions, the nilpotent doubling chain, Mu/Cst, outputs) runs ONCE on a
block-diagonal 128x128 matrix (MXU-shaped; cross-block entries are masked away, products of block-diagonal matrices stay block-diagonal), while the carried state is
stepped per 16-token block (2 short dots) from VMEM. One fused kernel per direction, zero-transpose "pair of heads" layout: the (B,T,H,N) tensors are read as
(B,T,H*N) and a 128-lane block holds a whole number of heads (two heads for N=64), so no wrapper transposes or padding exist.
Document isolation: the state is reset at document boundaries; padding rows (doc id >= PAD_DOC) are skipped and produce exact zeros.
"""
import functools, math
import jax, jax.numpy as jnp
from jax import lax
from ._common import (pl, pltpu, _bs, _call, _prec, _mm, _dot, _dot_nt, _dot_tn, _diff, _cum_mm, _keep, _shard_map, _data_parallel_ok, PAD_DOC)

def _tri_block(cb, L):
    """block-diagonal lower-triangular ones (cb*L, cb*L): ONE matmul gives the inclusive cumsum of all cb chunks of a block (K=cb*L fills the MXU depth)"""
    n = cb * L
    ri = lax.broadcasted_iota(jnp.int32, (n, n), 0); ci = lax.broadcasted_iota(jnp.int32, (n, n), 1)
    if L & (L - 1) == 0: sh = L.bit_length() - 1; same = (ri >> sh) == (ci >> sh)
    else: same = lax.div(ri, jnp.int32(L)) == lax.div(ci, jnp.int32(L))
    return ((ri >= ci) & same).astype(jnp.float32)


def seq_state(S, pre, prec):
    pb = prec[1] if isinstance(prec, tuple) else prec
    return S * pre[3] - _dot(S, pre[2], pb) + pre[4]

def seq_out(S, pre, prec):
    pb = prec[1] if isinstance(prec, tuple) else prec
    return _dot_nt(pre[0], S, pb) + pre[1]

def seq_bwd(S, pre, dY, dS, prec):
    """hand-written backward of (seq_out, seq_state): dY cotangent of Y, dS cotangent of S_next -> (dS_prev, cotangent of pre)"""
    pb = prec[1] if isinstance(prec, tuple) else prec
    Ra, _, Wk, eLk, _ = pre
    dRa = _dot(dY, S, pb)
    dS_prev = _dot_tn(dY, Ra, pb) + dS * eLk - _dot_nt(dS, Wk, pb)
    dWk = -_dot_tn(S, dS, pb); deLk = jnp.sum(dS * S, axis=0, keepdims=True)
    return dS_prev, (dRa, dY, dWk, deLk, dS)

def _ones_block(M, L):
    ri = lax.broadcasted_iota(jnp.int32, (M, M), 0); ci = lax.broadcasted_iota(jnp.int32, (M, M), 1); sh = L.bit_length() - 1
    return ((ri >> sh) == (ci >> sh)).astype(jnp.float32)

@functools.lru_cache(maxsize=None)
def _mk_inv_saved(ps):
    """identity on a saved inverse Minv = (I + Bc)^-1 whose vjp w.r.t. Bc is the exact closed form  dBc = -Minv^T G Minv^T  (G = cotangent of Minv): no recomputation of the doubling chain
    and no differentiation through it (backprop through 6 chained, cancellation-prone products is both slower and less accurate than this 2-dot formula)."""
    @jax.custom_vjp
    def f(Bc, Minv): return Minv
    def fwd(Bc, Minv): return Minv, Minv
    def bwd(Minv, G): return -_dot_nt(_dot_tn(Minv, G, ps), Minv, ps), jnp.zeros_like(Minv)
    f.defvjp(fwd, bwd); return f


def pre_packed(r, lw, lg, lgLf, k, v, kk, a, dcol, drow, dinc, dout, L, prec, minv=None, ret_minv=False):
    """state-independent algebra of G = M/L consecutive L-token blocks (M rows), packed. lg = per-block inclusive cumsum of lw, lgLf = per-block TOTAL log decay broadcast to every row
    of its block (both computed by the caller with exact block-diagonal matmuls). returns Ra, Ystat (M,N) and per-block tuples Wk (N,N), eL (1,N), Sst (N,N) with
    Y_c = Ra_c S_c^T + Ystat_c,  S_{c+1} = S_c * eL_c - S_c Wk_c + Sst_c  (document-carry factor already folded in)."""
    f32 = jnp.float32; ps, pb = prec if isinstance(prec, tuple) else (prec, prec)
    M, N = lw.shape; G = M // L
    ri = lax.broadcasted_iota(jnp.int32, (M, M), 0); ci = lax.broadcasted_iota(jnp.int32, (M, M), 1); sh = L.bit_length() - 1
    blk = (ri >> sh) == (ci >> sh); same = (dcol == drow).astype(f32)
    strict = (blk & (ri > ci)).astype(f32) * same; incl = (blk & (ri >= ci)).astype(f32) * same; I = (ri == ci).astype(f32)
    lgp = lg - lw; b = kk * a
    en = jnp.exp(-lg); wn = jnp.exp(lgLf) * en
    Up = kk * jnp.exp(lgp); Rp = r * jnp.exp(lg); Kp = k * en; Bp = b * en; Khat = k * wn; Bhat = b * wn
    if ps == pb:
        UR = jnp.concatenate([Up, Rp], axis=0); Z1 = _dot_nt(UR, Kp, ps); Z2 = _dot_nt(UR, Bp, ps)
        Kc = Z1[:M] * strict; Kr = Z1[M:] * incl; Bc = Z2[:M] * strict; Br = Z2[M:] * incl
    else:
        Kc = _dot_nt(Up, Kp, ps) * strict; Bc = _dot_nt(Up, Bp, ps) * strict; Kr = _dot_nt(Rp, Kp, pb) * incl; Br = _dot_nt(Rp, Bp, pb) * incl
    Minv, Pw = I - Bc, Bc                                                                   # block-diagonal, strictly lower blocks of order L: nilpotent, (I+Bc)^-1 = (I-Bc)(I+Bc^2)...(I+Bc^(L/2))
    if minv is not None: Minv = _mk_inv_saved(ps)(Bc, minv)                              # backward pass: the inverse saved by the forward kernel
    else:
        for _ in range(int(math.log2(L)) - 1): Pw = _dot(Pw, Pw, ps); Minv = _dot(Minv, I + Pw, ps)
    X = _dot(Minv, jnp.concatenate([Up, _dot(Kc, v, ps)], axis=1), ps); Mu, Cst = X[:, :N], X[:, N:]
    Z = _dot(Br, X, pb)
    cm = (dcol == dinc).astype(f32); om = (dcol == dout).astype(f32)
    Ra = (Rp - Z[:, :N]) * cm; Ystat = _dot(Kr, v, pb) - Z[:, N:]
    Wks, eLs, Ssts = [], [], []
    for c in range(G):
        sl = slice(c * L, (c + 1) * L)
        keep = (jnp.broadcast_to(dinc[sl], (L, N)) == jnp.broadcast_to(dout[sl], (L, N))).astype(f32)[0:1, :]
        Ssts.append(_dot_tn(jnp.concatenate([v[sl] * om[sl], Cst[sl] * om[sl]], axis=0), jnp.concatenate([Khat[sl], -Bhat[sl]], axis=0), pb))
        Wks.append(keep * _dot_tn(Mu[sl], Bhat[sl], pb)); eLs.append(keep * jnp.exp(lgLf[c * L:c * L + 1, :]))
    out = (Ra, Ystat, tuple(Wks), tuple(eLs), tuple(Ssts))
    return (out, Minv) if ret_minv else out

def _wk_args(refs, hh, N, M, L):
    r_ref, lw_ref, k_ref, v_ref, kk_ref, a_ref = refs; sl = slice(hh * N, (hh + 1) * N)
    return tuple(ref[0, :, sl] for ref in (r_ref, lw_ref, k_ref, v_ref, kk_ref, a_ref))

def _wk_pre_call(x, T, Tn, m, L, prec, ret_minv=False):
    r, lw, k, v, kk, a = x
    return pre_packed(r, lw, _cum_mm(T, lw), _cum_mm(Tn, lw), k, v, kk, a, *m, L, prec, ret_minv=ret_minv)

def _wk_fwd_kernel(nv_ref, r_ref, lw_ref, k_ref, v_ref, kk_ref, a_ref, dc_ref, dr_ref, di_ref, do_ref, y_ref, sst_ref, mv_ref, S_scr, *, M, L, N, hp, prec):
    b = pl.program_id(0); g = pl.program_id(2); G = M // L
    @pl.when(g == 0)
    def _():
        S_scr[...] = jnp.zeros_like(S_scr)
    @pl.when(g * M < nv_ref[b])
    def _():
        T = _tri_block(G, L); Tn = _ones_block(M, L); m = (dc_ref[0], dr_ref[0, 0], di_ref[0], do_ref[0])
        pm = [_wk_pre_call(_wk_args((r_ref, lw_ref, k_ref, v_ref, kk_ref, a_ref), hh, N, M, L), T, Tn, m, L, prec, ret_minv=True) for hh in range(hp)]
        pres = [x[0] for x in pm]
        for hh in range(hp): mv_ref[0, hh, 0] = pm[hh][1]                                   # (I + Bc)^-1 for the backward pass
        Ss = [S_scr[hh] for hh in range(hp)]
        for hh in range(hp): sst_ref[0, hh, 0] = Ss[hh]                                        # state at the start of the program (for the backward)
        for c in range(G):
            rows = slice(c * L, (c + 1) * L)
            for hh in range(hp):
                p = pres[hh]; y_ref[0, rows, hh * N:(hh + 1) * N] = seq_out(Ss[hh], (p[0][rows], p[1][rows], p[2][c], p[3][c], p[4][c]), prec)
            for hh in range(hp):
                p = pres[hh]; Ss[hh] = seq_state(Ss[hh], (p[0][rows], p[1][rows], p[2][c], p[3][c], p[4][c]), prec)
        for hh in range(hp): S_scr[hh] = Ss[hh]
    @pl.when(jnp.logical_not(g * M < nv_ref[b]))
    def _():
        y_ref[...] = jnp.zeros_like(y_ref); mv_ref[...] = jnp.zeros_like(mv_ref)
        for hh in range(hp): sst_ref[0, hh, 0] = S_scr[hh]

def _wk_bwd_kernel(nv_ref, r_ref, lw_ref, k_ref, v_ref, kk_ref, a_ref, dc_ref, dr_ref, di_ref, do_ref, dy_ref, sst_ref, mv_ref,
                   gr_ref, glw_ref, gk_ref, gv_ref, gkk_ref, ga_ref, dS_scr, *, M, L, N, hp, nG, prec):
    b = pl.program_id(0); j = pl.program_id(2); gg = nG - 1 - j; G = M // L                     # groups are visited in reverse order (index maps)
    @pl.when(j == 0)
    def _():
        dS_scr[...] = jnp.zeros_like(dS_scr)
    @pl.when(gg * M < nv_ref[b])
    def _():
        T = _tri_block(G, L); Tn = _ones_block(M, L); m = (dc_ref[0], dr_ref[0, 0], di_ref[0], do_ref[0]); vjs, pres = [], []
        for hh in range(hp):
            r, lw, k, v, kk, a = _wk_args((r_ref, lw_ref, k_ref, v_ref, kk_ref, a_ref), hh, N, M, L)
            lg = _cum_mm(T, lw); lgLf = _cum_mm(Tn, lw)
            with _diff(): out, vj = jax.vjp(lambda *t, mv=mv_ref[0, hh, 0]: pre_packed(*t, *m, L, prec, minv=mv), r, lw, lg, lgLf, k, v, kk, a)
            pres.append(out); vjs.append(vj)
        states = [[] for _ in range(hp)]; Ss = [sst_ref[0, hh, 0] for hh in range(hp)]
        for c in range(G):
            rows = slice(c * L, (c + 1) * L)
            for hh in range(hp): states[hh].append(Ss[hh])
            for hh in range(hp):
                p = pres[hh]; Ss[hh] = seq_state(Ss[hh], (p[0][rows], p[1][rows], p[2][c], p[3][c], p[4][c]), prec)
        dSs = [dS_scr[hh] for hh in range(hp)]
        dRa = [[None] * G for _ in range(hp)]; dWk = [[None] * G for _ in range(hp)]; deL = [[None] * G for _ in range(hp)]; dSt = [[None] * G for _ in range(hp)]
        for c in reversed(range(G)):
            rows = slice(c * L, (c + 1) * L)
            for hh in range(hp):
                p = pres[hh]
                dSs[hh], (dRa[hh][c], _, dWk[hh][c], deL[hh][c], dSt[hh][c]) = seq_bwd(states[hh][c], (p[0][rows], p[1][rows], p[2][c], p[3][c], p[4][c]), dy_ref[0, rows, hh * N:(hh + 1) * N], dSs[hh], prec)
        for hh in range(hp):
            sl = slice(hh * N, (hh + 1) * N)
            cot = (jnp.concatenate(dRa[hh], axis=0), dy_ref[0, :, sl], tuple(dWk[hh]), tuple(deL[hh]), tuple(dSt[hh]))
            gr, glw, glg, glgL, gk, gv, gkk, ga = vjs[hh](cot)
            gr_ref[0, :, sl] = gr; glw_ref[0, :, sl] = glw + _cum_mm(T, glg, 'tn') + _cum_mm(Tn, glgL, 'tn'); gk_ref[0, :, sl] = gk; gv_ref[0, :, sl] = gv; gkk_ref[0, :, sl] = gkk; ga_ref[0, :, sl] = ga
            dS_scr[hh] = dSs[hh]
    @pl.when(jnp.logical_not(gg * M < nv_ref[b]))
    def _():
        for ref in (gr_ref, glw_ref, gk_ref, gv_ref, gkk_ref, ga_ref): ref[...] = jnp.zeros_like(ref)

_HEADS_MULT = [1]                                   # experiment knob: heads per program = (128 // N) * this

@functools.lru_cache(maxsize=None)
def _make_core(H, N, M, L, prec_name, vmem_mb=None):
    prec = _prec(prec_name); hp = (max(1, 128 // N) if N < 128 else 1) * _HEADS_MULT[0]
    if H % hp or (hp * N) % 128: raise ValueError(f'rwkv7 needs head dim x heads-per-block to be a multiple of 128 (H={H}, N={N})')
    W = hp * N; P = H // hp
    def prep(doc, T):
        B = doc.shape[0]; nG = T // M; doc = doc.astype(jnp.int32); d16 = doc.reshape(B, T // L, L)
        d_in = jnp.concatenate([jnp.full((B, 1), -1, jnp.int32), d16[:, :-1, -1]], 1); d_out = d16[:, :, -1]                # per L-token block, carried across programs
        nv = jnp.max(jnp.where(doc < PAD_DOC, jnp.arange(T, dtype=jnp.int32)[None] + 1, 0), axis=1).astype(jnp.int32)
        return nv, (doc[..., None], doc.reshape(B, nG, 1, M), jnp.repeat(d_in, L, axis=1)[..., None], jnp.repeat(d_out, L, axis=1)[..., None])
    def specs(nG, rev):
        cm = (lambda c: nG - 1 - c) if rev else (lambda c: c)
        xs = _bs((1, M, W), lambda b, p, c, *_: (b, cm(c), p)); cs = _bs((1, M, 1), lambda b, p, c, *_: (b, cm(c), 0)); rs = _bs((1, 1, 1, M), lambda b, p, c, *_: (b, cm(c), 0, 0))
        nn = _bs((1, hp, 1, N, N), lambda b, p, c, *_: (b, p, cm(c), 0, 0)); mm = _bs((1, hp, 1, M, M), lambda b, p, c, *_: (b, p, cm(c), 0, 0))
        return xs, cs, rs, nn, mm
    def fwd_impl(r, lw, k, v, kk, a, doc):
        B, T = r.shape[:2]; nG = T // M; nv, ms = prep(doc, T); flat = [x.reshape(B, T, H * N) for x in (r, lw, k, v, kk, a)]
        xs, cs, rs, nn, mm = specs(nG, False); f32 = jnp.float32; sds = jax.ShapeDtypeStruct
        y, sst, mv = _call(functools.partial(_wk_fwd_kernel, M=M, L=L, N=N, hp=hp, prec=prec), (B, P, nG), [xs] * 6 + [cs, rs, cs, cs], [xs, nn, mm],
                       [sds((B, T, H * N), f32), sds((B, H, nG, N, N), f32), sds((B, H, nG, M, M), f32)], [pltpu.VMEM((hp, N, N), f32)], ('parallel', 'parallel', 'arbitrary'), nsp=1, vmem_mb=vmem_mb)(nv, *flat, *ms)
        return y.reshape(B, T, H, N), (sst, mv)
    @jax.custom_vjp
    def core(r, lw, k, v, kk, a, doc): return fwd_impl(r, lw, k, v, kk, a, doc)[0]
    def core_fwd(r, lw, k, v, kk, a, doc):
        y, (sst, mv) = fwd_impl(r, lw, k, v, kk, a, doc); y, sst, mv = _keep(y, sst, mv); return y, (r, lw, k, v, kk, a, doc, sst, mv)
    def core_bwd(res, dy):
        r, lw, k, v, kk, a, doc, sst, mv = res
        B, T = r.shape[:2]; nG = T // M; nv, ms = prep(doc, T); flat = [x.reshape(B, T, H * N) for x in (r, lw, k, v, kk, a)]; dyf = dy.reshape(B, T, H * N)
        xs, cs, rs, nn, mm = specs(nG, True); f32 = jnp.float32; sds = jax.ShapeDtypeStruct
        outs = _call(functools.partial(_wk_bwd_kernel, M=M, L=L, N=N, hp=hp, nG=nG, prec=prec), (B, P, nG), [xs] * 6 + [cs, rs, cs, cs, xs, nn, mm], [xs] * 6,
                     [sds((B, T, H * N), f32)] * 6, [pltpu.VMEM((hp, N, N), f32)], ('parallel', 'parallel', 'arbitrary'), nsp=1, vmem_mb=vmem_mb)(nv, *flat, *ms, dyf, sst, mv)
        return (*[o.reshape(B, T, H, N) for o in outs], None)
    core.defvjp(core_fwd, core_bwd)
    return core

def make_rwkv7(precision='mixed', M=128, L=16, vmem_mb=None):
    """Build the kernel (see the module docstring). Solve blocks of L=16 tokens (numerically safe), programs of M tokens (largest of M, M/2, ... that divides T)."""
    def fn(r, lw, k, v, kk, a, L_=16, doc=None):
        B, T, H, N = r.shape
        if doc is None: doc = jnp.zeros((B, T), jnp.int32)
        Mc = M
        while Mc > L and T % Mc: Mc //= 2
        if T % Mc: raise ValueError(f'rwkv7: T={T} is not a multiple of {L}')
        core = _make_core(H, N, Mc, L, precision, vmem_mb)
        if _data_parallel_ok(B):
            from jax.sharding import PartitionSpec as P
            sp = P('data'); return _shard_map(lambda *xs: core(*xs), (sp,) * 7, sp)(r, lw, k, v, kk, a, doc)
        return core(r, lw, k, v, kk, a, doc)
    return fn


