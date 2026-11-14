"""VeroKernel 1.0 - fused Pallas TPU kernels for RWKV-7 and Transformer (flash) attention, with Rust host-side helpers.

`import verokernel` is instant: jax and the kernels are imported lazily on first use of rwkv7 / attention / autotune.
Host-side helpers (document ids, block tables, CPU reference kernels) never import jax.
"""
from ._version import __version__
from .host import (PAD_DOC, NATIVE, block_doc_ranges, valid_length, doc_ids_from_lengths, doc_ids_from_starts, rwkv7_cpu, attention_cpu, num_threads)

_LAZY = {
    'rwkv7': 'ops', 'attention': 'ops', 'rwkv7_supported': 'ops', 'attention_supported': 'ops',
    'set_mesh': '_common', 'get_mesh': '_common',
    'autotune': 'autotune', 'reference': None,
}

__all__ = ['__version__', 'PAD_DOC', 'NATIVE', 'block_doc_ranges', 'valid_length', 'doc_ids_from_lengths', 'doc_ids_from_starts', 'rwkv7_cpu', 'attention_cpu',
           'rwkv7', 'attention', 'rwkv7_supported', 'attention_supported', 'set_mesh', 'get_mesh', 'autotune', 'reference', 'info']


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod = _LAZY[name]
        if mod is None: return importlib.import_module(f'.{name}', __name__)
        m = importlib.import_module(f'.{mod}', __name__)
        val = getattr(m, name) if name != 'autotune' else m.autotune
        globals()[name] = val
        return val
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def info():
    """Short description of the installation (does import jax)."""
    import jax
    return dict(version=__version__, native_extension=NATIVE, threads=num_threads(), jax=jax.__version__, backend=jax.default_backend(),
                devices=[d.device_kind for d in jax.devices()][:8])
