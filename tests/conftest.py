import importlib.util, os, sys
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
# Use the installed package (wheel, with the Rust extension) when there is one; otherwise fall back to the source tree (NumPy fallback, no Rust).
if importlib.util.find_spec('verokernel') is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
