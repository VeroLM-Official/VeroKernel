# Changelog

## 1.0.0
First public release.
- RWKV-7 fused forward/backward Pallas TPU kernel (16-token solve blocks packed into 128-token programs, document-isolated state, padding skipping).
- Document/validity-masked flash attention (forward, dq, dkv), causal or bidirectional, block skipping by document ranges.
- Exact XLA fallbacks for every kernel; `backend='auto'` picks Pallas on TPU and XLA elsewhere.
- Rust extension: document-id / block-table preparation, multi-threaded CPU reference kernels for RWKV-7 and attention. Pure-NumPy fallback when the extension is not installed.
- Lazy imports: `import verokernel` does not import jax.
