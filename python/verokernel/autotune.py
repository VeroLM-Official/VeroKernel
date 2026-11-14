"""Optional on-device selection between the Pallas kernels and the XLA fallback.

`autotune(B, T, H, N, ...)` times every candidate on the CURRENT device with a correlated-key stress test as accuracy gate and returns the fastest accurate one.
Crash-safe: a candidate that killed the process (a Mosaic crash) is written to the cache file beforehand and skipped on the next run.
Most users do not need it: backend='auto' already picks Pallas on TPU. Use it to choose between precision='mixed' / 'high' and attention block sizes for your shapes.
"""
import json, os, time, hashlib, math
import numpy as np
import jax, jax.numpy as jnp
from . import reference as ref
from . import ops
from ._common import PAD_DOC


def _rel(a, b):
    a = jnp.asarray(a, jnp.float32); b = jnp.asarray(b, jnp.float32)
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-30))


def _time(fn, args, iters):
    jax.block_until_ready(fn(*args)); t0 = time.perf_counter()
    for _ in range(iters): out = fn(*args)
    jax.block_until_ready(out); return (time.perf_counter() - t0) / iters * 1e3


def rwkv7_inputs(B, T, H, N, corr=0.0, seed=0):
    """Test inputs with packed documents and ~20% padding. corr in [0,1]: correlation of neighbouring keys (1.0 = repeated text; this is what breaks long-chunk solves)."""
    rng = np.random.default_rng(seed); sg = lambda x: 1 / (1 + np.exp(-x))
    r = rng.normal(size=(B, T, H, N)); k = rng.normal(size=(B, T, H, N)) * 0.5; v = rng.normal(size=(B, T, H, N))
    lw = -sg(rng.normal(size=(B, T, H, N))) * math.exp(-0.5)
    kk = corr * rng.normal(size=(B, 1, H, N)) + (1 - corr) * rng.normal(size=(B, T, H, N)); kk /= np.linalg.norm(kk, axis=-1, keepdims=True)
    a = sg(rng.normal(size=(B, T, H, N))) if corr < 0.9 else np.full((B, T, H, N), 0.95)
    doc = np.zeros((B, T), np.int32)
    for b in range(B):
        nv = T - int(T * 0.2) - b * 3
        for c in np.sort(rng.choice(np.arange(5, nv - 5), max(2, T // 96), replace=False)): doc[b, c:] += 1
        doc[b, nv:] = PAD_DOC
    w = rng.normal(size=(B, T, H, N)) * (doc < PAD_DOC)[:, :, None, None]
    xs = [jnp.asarray(x.astype(np.float32)) for x in (r, lw, k, v, kk, a)]
    return xs, jnp.asarray(doc), jnp.asarray(w.astype(np.float32))


def _eval_rwkv7(fn, B, T, H, N, corr, iters):
    xs, doc, w = rwkv7_inputs(B, T, H, N, corr)
    vm = (doc < PAD_DOC)[:, :, None, None]
    with jax.default_matmul_precision('highest'):
        ry = ref.wkv7_chunked(*xs, 16, doc=doc); rg = jax.grad(lambda *x: jnp.sum(ref.wkv7_chunked(*x, 16, doc=doc) * w), argnums=tuple(range(6)))(*xs)
    call = lambda *x: fn(*x, doc)
    vg = jax.jit(jax.value_and_grad(lambda *x: jnp.sum(call(*x) * w), argnums=tuple(range(6))))
    y = jax.jit(call)(*xs); _, g = vg(*xs)
    if not (bool(jnp.all(jnp.isfinite(jnp.where(vm, y, 0)))) and all(bool(jnp.all(jnp.isfinite(t))) for t in g)):
        return dict(status='non-finite')
    return dict(status='ok', fwd_err=_rel(jnp.where(vm, y, 0), jnp.where(vm, ry, 0)), grad_err=max(_rel(a, b) for a, b in zip(g, rg)), ms=_time(vg, xs, iters))


def autotune(B, T, H, N, cache_path=None, iters=4, log=print, fwd_tol=2e-2, grad_tol=6e-2):
    """Time RWKV-7 candidates for shape (B, T, H, N) on this device. Returns dict(best=<name>, report=...). Candidate names: 'pallas_mixed', 'pallas_high', 'xla'.
    Every candidate must pass the correlated-key stress test (finite, forward error <= fwd_tol, gradient error <= grad_tol vs the float32/highest-precision reference)."""
    cands = {'xla': lambda *x: ref.wkv7_chunked(*x[:6], 16, doc=x[6]) * (x[6] < PAD_DOC)[:, :, None, None]}
    if jax.default_backend() == 'tpu' and ops.rwkv7_supported((B, T, H, N))[0]:
        from ._rwkv7_kernel import make_rwkv7
        for pr in ('mixed', 'high'):
            f = make_rwkv7(pr); cands[f'pallas_{pr}'] = (lambda f: lambda *x: f(*x[:6], 16, x[6]))(f)
    key = dict(jax=jax.__version__, backend=jax.default_backend(), dev=str(jax.devices()[0].device_kind), shape=[B, T, H, N], src=_src_hash())
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try: cache = json.load(open(cache_path))
        except Exception: cache = {}
    if cache.get('key') == key and cache.get('best'):
        log(f'[verokernel] cached choice: {cache["best"]}'); return dict(best=cache['best'], report=cache.get('report', {}))
    bad = set(cache.get('bad', []))
    if cache.get('inflight'): bad.add(cache['inflight']); log(f'[verokernel] candidate {cache["inflight"]} killed the previous process -> blacklisted')
    def save():
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            tmp = cache_path + '.tmp'; json.dump(cache, open(tmp, 'w')); os.replace(tmp, cache_path)
    cache.update(bad=sorted(bad), inflight=None); save()
    report = {}
    for name, fn in cands.items():
        if name in bad: report[name] = dict(status='blacklisted'); continue
        cache['inflight'] = name; save()
        try:
            r = None
            for corr in (0.0, 0.98, 1.0):                                      # random keys, then the correlated-key stress sets
                r = _eval_rwkv7(fn, B, T, H, N, corr, iters)
                if r['status'] != 'ok': break
                if r['fwd_err'] > fwd_tol or r['grad_err'] > grad_tol: r['status'] = f'inaccurate at key-correlation {corr}'; break
        except Exception as e:
            r = dict(status=f'failed: {type(e).__name__}: {str(e)[:160]}')
        cache['inflight'] = None; save(); report[name] = r
        log(f'[verokernel] rwkv7 {name:13s} ' + (f'{r["ms"]:8.2f} ms fwd+bwd  fwd_err {r["fwd_err"]:.1e} grad_err {r["grad_err"]:.1e}' if r['status'] == 'ok' else r['status']))
    ok = {n: r for n, r in report.items() if r.get('status') == 'ok'}
    best = min(ok, key=lambda n: ok[n]['ms']) if ok else 'xla'
    cache.update(key=key, best=best, report=report); save(); log(f'[verokernel] CHOICE rwkv7={best}')
    return dict(best=best, report=report)


def _src_hash():
    h = hashlib.sha256()
    d = os.path.dirname(__file__)
    for n in ('_rwkv7_kernel.py', '_flash_kernel.py', '_common.py'):
        h.update(open(os.path.join(d, n), 'rb').read())
    return h.hexdigest()[:16]
