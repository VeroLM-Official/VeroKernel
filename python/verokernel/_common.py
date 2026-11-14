"""Shared plumbing for the Pallas kernels: mesh handling, pallas_call wrapper, precision-controlled dots.

Imported lazily by verokernel (importing it pulls in jax and pallas).
"""
import functools, math
import numpy as np
import jax, jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
try:
    from jax.experimental.pallas import tpu as pltpu
except Exception:                                                                    # pragma: no cover
    pltpu = None

NEG = -1e30
PAD_DOC = 1 << 28          # document id that marks padding rows for the RWKV-7 kernel (rows with id >= PAD_DOC are skipped and come back as zeros)
_MESH = None


def set_mesh(mesh):
    """Data-parallel mesh (axis name 'data'). Pallas calls are wrapped in shard_map over the batch axis so XLA never gathers the batch onto one chip."""
    global _MESH; _MESH = mesh


def get_mesh(): return _MESH


_FORCE_TPU_LOWERING = False        # tests: lower for TPU on a CPU host (checks the Mosaic lowering rules, no execution)
def _interpret(): return False if _FORCE_TPU_LOWERING else jax.default_backend() != 'tpu'

def _bs(shape, fn): return pl.BlockSpec(block_shape=shape, index_map=fn)

def _cparams(sem, vmem_mb=None):
    cls = getattr(pltpu, 'CompilerParams', None) or getattr(pltpu, 'TPUCompilerParams', None)
    if cls is None: return None
    return cls(dimension_semantics=sem, vmem_limit_bytes=int(vmem_mb * 2 ** 20)) if vmem_mb else cls(dimension_semantics=sem)

def _call(kernel, grid, in_specs, out_specs, out_shape, scratch, sem, nsp=0, vmem_mb=None):
    interpret = _interpret()
    gs = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=nsp, grid=grid, in_specs=in_specs, out_specs=out_specs, scratch_shapes=scratch)
    kw = dict(grid_spec=gs, out_shape=out_shape, interpret=interpret)
    if not interpret:
        cp = _cparams(sem, vmem_mb)
        if cp is not None: kw['compiler_params'] = cp
    return pl.pallas_call(kernel, **kw)

def _shard_map(fn, in_specs, out_specs):
    from jax.sharding import PartitionSpec as P
    sm = getattr(jax, 'shard_map', None)
    if sm is None:
        from jax.experimental.shard_map import shard_map as sm
    for kw in ({'check_vma': False}, {'check_rep': False}, {}):
        try: return sm(fn, mesh=_MESH, in_specs=in_specs, out_specs=out_specs, **kw)
        except TypeError: continue
    raise RuntimeError('shard_map unavailable')

def _data_parallel_ok(B):
    return _MESH is not None and _MESH.shape.get('data', 1) > 1 and B % _MESH.shape['data'] == 0

X3 = 'x3'      # Mosaic only supports DEFAULT (1 bf16 pass) and HIGHEST (fp32, 6 passes) dot precisions, NOT HIGH: the 3-pass variant is written out by hand (bf16 hi/lo split, 3 dots)
def _prec(name):
    return {'highest': lax.Precision.HIGHEST, 'high': X3, 'default': None, 'mixed': (X3, None)}[name]   # mixed = (sensitive dots, bulk dots)
# ============================================================================= RWKV-7 chunk algebra (pure jnp: used inside the kernels and by jax.vjp for the backward)
_NN = (((1,), (0,)), ((), ())); _NT_ = (((1,), (1,)), ((), ())); _TN_ = (((0,), (0,)), ((), ()))
def _mm(a, b, dn, prec=None): return lax.dot_general(a, b, dn, precision=prec, preferred_element_type=jnp.float32)
def _split(x):
    hi = x.astype(jnp.bfloat16); return hi, (x - hi.astype(jnp.float32)).astype(jnp.bfloat16)
def _x3_plain(a, b, dn):
    """~fp32-accurate product from 3 single-pass bf16 dots: hi*hi + hi*lo + lo*hi"""
    ah, al = _split(a); bh, bl = _split(b)
    return _mm(ah, bh, dn) + _mm(ah, bl, dn) + _mm(al, bh, dn)
_DIFF = [False]        # True while jax.vjp traces a kernel body: then the hand-written 3-pass dots use a custom VJP (the casts of the hi/lo split must not be differentiated)
class _diff:
    def __enter__(self): _DIFF[0] = True
    def __exit__(self, *a): _DIFF[0] = False
def _mk_x3_cv(kind):
    dn = {'nn': _NN, 'nt': _NT_, 'tn': _TN_}[kind]
    @jax.custom_vjp
    def f(a, b): return _x3_plain(a, b, dn)
    def fwd(a, b): return _x3_plain(a, b, dn), (a, b)
    def bwd(res, g):
        a, b = res
        if kind == 'nn': return _x3_plain(g, b, _NT_), _x3_plain(a, g, _TN_)
        if kind == 'nt': return _x3_plain(g, b, _NN), _x3_plain(g, a, _TN_)
        return _x3_plain(b, g, _NT_), _x3_plain(a, g, _NN)
    f.defvjp(fwd, bwd); return f
_X3_CV = {k: _mk_x3_cv(k) for k in ('nn', 'nt', 'tn')}
def _dg(a, b, kind, prec):
    if isinstance(prec, str) and prec == X3:
        return _X3_CV[kind](a, b) if _DIFF[0] else _x3_plain(a, b, {'nn': _NN, 'nt': _NT_, 'tn': _TN_}[kind])
    return _mm(a, b, {'nn': _NN, 'nt': _NT_, 'tn': _TN_}[kind], prec)
def _dot(a, b, prec): return _dg(a, b, 'nn', prec)
def _dot_nt(a, b, prec): return _dg(a, b, 'nt', prec)       # a @ b.T
def _dot_tn(a, b, prec): return _dg(a, b, 'tn', prec)       # a.T @ b

def _cum_mm(T, x, kind='nn'):
    """T (0/1 matrix, exact in bf16) times an fp32 x, fp32-exact with 3 bf16 passes instead of HIGHEST's 6 (not differentiated: used outside jax.vjp)"""
    Tb = T.astype(jnp.bfloat16); h = x.astype(jnp.bfloat16); r = x - h.astype(jnp.float32); m = r.astype(jnp.bfloat16); l = (r - m.astype(jnp.float32)).astype(jnp.bfloat16)
    dn = _NN if kind == 'nn' else _TN_
    return _mm(Tb, h, dn) + _mm(Tb, m, dn) + _mm(Tb, l, dn)
def _keep(*xs):
    """names the saved-for-backward outputs of a fused wkv kernel 'wkv_res': only the 'dots_nb_wkv' remat policy keeps them (so the backward pass does not re-run the forward kernel); identity otherwise"""
    from jax.ad_checkpoint import checkpoint_name
    out = tuple(checkpoint_name(x, 'wkv_res') for x in xs); return out if len(out) > 1 else out[0]

