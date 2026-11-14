"""Pure-XLA reference implementations (any backend). They define the exact semantics the Pallas kernels are verified against,
and they are the automatic fallback on CPU / GPU or when a kernel's shape limits are not met.
"""
import math
import jax, jax.numpy as jnp
from jax import lax


def wkv7_sequential(r, lw, k, v, kk, a, doc=None):
    """The recurrence itself (token by token, lax.scan); all (B,T,H,N) fp32. doc (B,T) int: the state is reset before a token whose document id differs from the previous one."""
    B, T, H, N = r.shape
    ds = jnp.zeros((B, T), bool) if doc is None else jnp.concatenate([jnp.zeros((B, 1), bool), doc[:, 1:] != doc[:, :-1]], axis=1)
    def step(S, inp):
        r_t, lw_t, k_t, v_t, kk_t, a_t, d_t = inp
        S = jnp.where(d_t[:, None, None, None], 0.0, S)
        Skk = jnp.einsum('bhij,bhj->bhi', S, kk_t)
        S = S * jnp.exp(lw_t)[:, :, None, :] - Skk[..., None] * (kk_t * a_t)[:, :, None, :] + v_t[..., None] * k_t[:, :, None, :]
        return S, jnp.einsum('bhij,bhj->bhi', S, r_t)
    xs = tuple(jnp.moveaxis(t, 1, 0) for t in (r, lw, k, v, kk, a)) + (jnp.moveaxis(ds, 1, 0),)
    _, ys = lax.scan(step, jnp.zeros((B, H, N, N), jnp.float32), xs)
    return jnp.moveaxis(ys, 0, 1)

def wkv7_chunked(r, lw, k, v, kk, a, L=16, doc=None):
    """chunk-parallel RWKV-7 (diag + rank-1 transition): sequential depth T/L instead of T.
    doc (B,T) int document ids (non-decreasing): the state is reset at document starts. Implemented as (a) masking every intra-chunk interaction
    matrix with the same-document mask (which also keeps (I+Bc)^-1 block-diagonal), (b) hiding the carried chunk state from tokens of a later
    document, (c) dropping the carried state / keeping only last-document tokens in the chunk summary."""
    B, T, H, N = r.shape; nC = T // L; assert nC * L == T
    def rs(x): return jnp.moveaxis(x.reshape(B, nC, L, H, N), 3, 1)                      # (B,H,nC,L,N)
    r, lw, k, v, kk, a = map(rs, (r, lw, k, v, kk, a))
    u, b = kk, kk * a
    lg = jnp.cumsum(lw, axis=3); lgp = lg - lw; lgL = lg[:, :, :, -1:, :]
    Up = u * jnp.exp(lgp); Rp = r * jnp.exp(lg); Kp = k * jnp.exp(-lg); Bp = b * jnp.exp(-lg)
    Khat = k * jnp.exp(lgL - lg); Bhat = b * jnp.exp(lgL - lg)
    mm = lambda A, Bm: jnp.einsum('...ln,...sn->...ls', A, Bm)
    strict = jnp.tril(jnp.ones((L, L), jnp.float32), -1); incl = jnp.tril(jnp.ones((L, L), jnp.float32))
    cm = om = keep = None
    if doc is not None:
        dch = doc.reshape(B, nC, L); dc = dch[:, None]                                  # (B,1,nC,L)
        same = (dc[..., :, None] == dc[..., None, :]).astype(jnp.float32)               # (B,1,nC,L,L)
        strict, incl = strict * same, incl * same
        d_in = jnp.concatenate([jnp.full((B, 1), -1, dch.dtype), dch[:, :-1, -1]], 1)    # document owning the carried state of every chunk
        d_out = dch[:, :, -1]
        cm = (dc == d_in[:, None, :, None]).astype(jnp.float32)[..., None]               # tokens that may read the carried state
        om = (dc == d_out[:, None, :, None]).astype(jnp.float32)[..., None]              # tokens that belong to the last document of the chunk
        keep = (d_in == d_out).astype(jnp.float32)[:, None, :, None, None]               # the carried state survives the whole chunk
    if L >= 32:                                                                           # the four interaction matrices as ONE (2L x N) @ (N x 2L) matmul (MXU-sized for L >= 64)
        Z = mm(jnp.concatenate([Up, Rp], 3), jnp.concatenate([Kp, Bp], 3))
        Kc, Bc, Kr, Br = Z[..., :L, :L] * strict, Z[..., :L, L:] * strict, Z[..., L:, :L] * incl, Z[..., L:, L:] * incl
    else: Kc, Bc, Kr, Br = mm(Up, Kp) * strict, mm(Up, Bp) * strict, mm(Rp, Kp) * incl, mm(Rp, Bp) * incl
    mat = lambda A, Bm: jnp.einsum('...ij,...jk->...ik', A, Bm)
    I = jnp.eye(L, dtype=jnp.float32)
    assert L & (L - 1) == 0, 'chunk length must be a power of two'
    Minv, Pw = I - Bc, Bc                                                                 # Bc is nilpotent (strictly lower, order L): (I+Bc)^-1 = (I-Bc)(I+Bc^2)(I+Bc^4)...(I+Bc^(L/2))
    for _ in range(int(math.log2(L)) - 1): Pw = mat(Pw, Pw); Minv = mat(Minv, I + Pw)
    Mu = mat(Minv, Up); Cstat = mat(Minv, mat(Kc, v))
    Ra = Rp - mat(Br, Mu); Ystat = mat(Kr, v) - mat(Br, Cstat)
    expL = jnp.exp(lgL[:, :, :, 0, :])                                                   # (B,H,nC,N)
    Ach = expL[..., :, None] * jnp.eye(N, dtype=jnp.float32) - jnp.einsum('...ln,...lm->...nm', Mu, Bhat)
    vs, Cs = (v, Cstat) if om is None else (v * om, Cstat * om)
    Sst = jnp.einsum('...lv,...lk->...vk', vs, Khat) - jnp.einsum('...lv,...lk->...vk', Cs, Bhat)
    if keep is not None: Ach = Ach * keep
    def stepf(S, inp):
        Ac, Ss = inp
        return S @ Ac + Ss, S
    xs = (jnp.moveaxis(Ach, 2, 0), jnp.moveaxis(Sst, 2, 0))
    _, Sstart = lax.scan(stepf, jnp.zeros((B, H, N, N), jnp.float32), xs)
    Sstart = jnp.moveaxis(Sstart, 0, 2)                                                  # (B,H,nC,N,N): state at the start of each chunk
    if cm is not None: Ra = Ra * cm
    Y = jnp.einsum('...lk,...vk->...lv', Ra, Sstart) + Ystat
    return jnp.moveaxis(Y, 1, 3).reshape(B, T, H, N)

def attention_xla(q, k, v, doc, valid, causal=True, block=128):
    """XLA implementation of the masked attention core (query-blocked, rematerialised per block).
    q,k,v (B,H,T,hd); doc (B,T) int; valid (B,T) bool. Returns (o (B,H,T,hd), max attention logit)."""
    B, H, T, hd = q.shape
    def blk(qi, ki, vi, dq, dk, vk, rows, cols):
        sc = jnp.einsum('bhqd,bhkd->bhqk', qi, ki).astype(jnp.float32) / math.sqrt(hd)
        m = (dq[:, :, None] == dk[:, None, :]) & vk[:, None, :]
        if causal: m = m & (rows[:, None] >= cols[None, :])[None]
        sc = jnp.where(m[:, None], sc, -1e30)
        mx = jnp.max(jnp.where(m[:, None], sc, -1e4))
        att = jax.nn.softmax(sc, -1).astype(q.dtype)
        has = jnp.any(m, -1)[:, None, :, None].astype(q.dtype)                    # queries without any attendable key output zeros
        return jnp.einsum('bhqk,bhkd->bhqd', att, vi) * has, mx
    outs, mxs = [], []
    for s0 in range(0, T, block):
        e0 = min(s0 + block, T); k_end = e0 if causal else T
        o, mx = jax.checkpoint(blk)(q[:, :, s0:e0], k[:, :, :k_end], v[:, :, :k_end], doc[:, s0:e0], doc[:, :k_end], valid[:, :k_end], jnp.arange(s0, e0), jnp.arange(k_end))
        outs.append(o); mxs.append(mx)
    return jnp.concatenate(outs, 2), jnp.max(jnp.stack(mxs))


def attention_dense(q, k, v, doc, valid, causal=True):
    """Dense float32 reference (materialises the (T,T) scores): the ground truth used by the tests."""
    B, H, T, hd = q.shape
    f = lambda t: t.astype(jnp.float32)
    sc = jnp.einsum('bhqd,bhkd->bhqk', f(q), f(k)) / math.sqrt(hd)
    m = (doc[:, :, None] == doc[:, None, :]) & valid[:, None, :]
    if causal: m = m & (jnp.arange(T)[:, None] >= jnp.arange(T)[None, :])[None]
    m = m[:, None]
    p = jax.nn.softmax(jnp.where(m, sc, -1e30), -1) * jnp.any(m, -1, keepdims=True)
    return jnp.einsum('bhqk,bhkd->bhqd', p, f(v))
