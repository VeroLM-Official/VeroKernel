# VeroKernel 1.0

Fused **Pallas TPU kernels** for **RWKV-7** and **Transformer (flash) attention**, with a **Rust** extension for host-side data preparation and CPU reference kernels.

*Türkçe: [README.tr.md](README.tr.md)*

```python
import verokernel as vk

y = vk.rwkv7(r, lw, k, v, kk, a, doc=doc)                 # RWKV-7 token mixing, (B,T,H,N), differentiable
o = vk.attention(q, k, v, doc, valid, causal=True)         # flash attention, (B,H,T,hd), differentiable
```

Pure JAX functions: they work with `jit`, `grad`, `remat`, Flax / Equinox / Optax, and nothing else is required. On TPU they run the Pallas kernels, everywhere else they fall back to exact XLA implementations automatically (`backend='auto'`).

## Speed

> In the author's own model training, switching to these kernels gave roughly **3x faster training** (end-to-end). This is a result from one model on one setup (TPU v5e), not a general guarantee: the gain depends on sequence length, head size, batch size and how much of your step time the token mixer takes. Run `benchmarks/bench.py` on your hardware to measure your own case.

## Features

**RWKV-7 kernel**
- Fused forward **and** hand-written backward kernels (no autodiff through the solve chain), one `pallas_call` per direction.
- Numerically safe by construction: 16-token triangular-solve blocks (longer blocks lose all digits when neighbouring keys are alike, e.g. repeated text), packed 8x into 128-token programs so every dense op is a 128x128 MXU-shaped matmul.
- **Document isolation**: several documents packed into one row never leak state into each other (state reset where the document id changes).
- **Padding skipping**: whole padding programs are not computed; padding rows always return exact zeros.
- Zero-transpose layout: `(B,T,H,N)` is read as `(B,T,H*N)` with a 128-lane block holding whole heads; no wrapper transposes or padding copies.
- Backward reuses the saved inverse of the solve matrix (closed-form gradient), saving the forward result with `checkpoint_name('wkv_res')` so a remat policy can avoid re-running the forward kernel.

**Attention kernel**
- Flash attention forward + `dq` + `dk/dv` kernels, bf16 or float32, no `(T,T)` score matrix.
- Document mask + key-validity mask, **causal or bidirectional**.
- **Block skipping**: (q-block, k-block) pairs of unrelated documents, blocks that are entirely padding, and (causal) future blocks cost nothing.
- Optional max-logit output (`return_max_logit=True`) for QK-clip style monitoring.

**Rust extension** (`verokernel._native`, optional but included in the wheels)
- `doc_ids_from_lengths`, `doc_ids_from_starts`, `valid_length`, `block_doc_ranges`: data-loader side preparation of the tables the kernels use.
- `rwkv7_cpu`, `attention_cpu`: multi-threaded float32 CPU reference kernels (forward). Useful for verification, debugging on a laptop, and tiny CPU runs.
- Falls back to NumPy automatically when the extension is not installed; `verokernel.NATIVE` tells you which one you have.

**Other**
- `import verokernel` is instant and does not import jax; jax and the kernels load on first use.
- `verokernel.autotune(...)`: optional on-device timing of Pallas vs XLA with a correlated-key accuracy gate, crash-safe cache.
- `verokernel.reference`: the exact XLA implementations (`wkv7_sequential`, `wkv7_chunked`, `attention_dense`, `attention_xla`) the kernels are tested against.

## Install

```bash
# 1) wheel from the GitHub Releases page (Linux / macOS / Windows, Rust extension included, no Rust toolchain needed)
pip install verokernel-1.0.0-<platform>.whl

# 2) straight from the repository (needs a Rust toolchain: https://rustup.rs)
pip install git+https://github.com/<user>/<repo>.git

# 3) development install
pip install maturin && maturin develop --release && pytest tests -q

# on a TPU VM, install JAX with TPU support first
pip install "jax[tpu]>=0.11.2" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Without the Rust extension (a plain copy of `python/verokernel`), everything still works through the NumPy fallback.

Releases: push a tag such as `1.0.0` and the `build` workflow publishes the wheels and the source distribution as a GitHub Release.

## API

```python
vk.rwkv7(r, lw, k, v, kk, a, doc=None, *, backend='auto', precision='mixed')
```
| argument | meaning |
|---|---|
| `r, lw, k, v, kk, a` | `(B,T,H,N)` tensors, cast to float32. `lw` = log decay (`<= 0`), `kk` = L2-normalised removal key, `a` = in-context learning rate |
| `doc` | `(B,T)` int32 document ids; state resets where the id changes; ids `>= vk.PAD_DOC` mark padding rows (zero output) |
| `backend` | `'auto'` (Pallas on TPU if the shape is supported, else XLA), `'pallas'` (raises if unsupported; runs in slow interpret mode off-TPU), `'xla'` |
| `precision` | `'mixed'` (default) or `'high'` (all dots 3-pass bf16, slower) |

Recurrence per head (state `S` is `N x N`, zero at the start of every document):
`S <- S * exp(lw)[None,:] - (S @ kk)[:,None] * (kk * a)[None,:] + v[:,None] * k[None,:]`, `y = S @ r`.

```python
vk.attention(q, k, v, doc=None, valid=None, *, causal=True, block=128, backend='auto', return_max_logit=False)
```
`q, k, v`: `(B,H,T,hd)`; logits are `q.k / sqrt(hd)` (apply qk-norm / positional encoding yourself). A query attends to keys of its own document with `valid=True` (and `key_pos <= query_pos` if causal). Queries with nothing to attend to return zeros.

Data parallel on a TPU mesh: `vk.set_mesh(mesh)` with a mesh that has a `'data'` axis wraps the kernels in `shard_map` over the batch axis.

## Limits

**RWKV-7**
- Fast path is **TPU only**. CPU/GPU use the exact XLA fallback (slower). There is no GPU/Triton kernel.
- Head size `N` must divide 128 or be a multiple of 128, and the number of heads must be a multiple of `128/N` (e.g. N=64 -> even `H`; N=32 -> `H % 4 == 0`). Check with `vk.rwkv7_supported((B,T,H,N))`; `backend='auto'` falls back to XLA otherwise.
- Float32 kernel (inputs are cast, output is float32).
- No initial-state input and no final-state output: every call starts from a zero state. It is a **training / full-sequence** kernel, not a stateful incremental decoder.
- `T` is padded internally to a multiple of 16; multiples of 128 are fastest (otherwise the program size shrinks to 64/32/16).
- Decay range: `lw` must be `<= 0`. Inside any 16-token block the summed `|lw|` must stay below about 88 (float32 `exp` overflow). In float32 the kernel was verified finite and exact to ~5e-7 for a constant per-token `|lw|` up to 5 and became NaN at 6. The reference training setup used `lw` in `[-0.6065, 0)` (decay in `(0.545, 1)`), far inside the safe range.
- Padding rows are best placed at the end of the row (that is what is skipped); padding in the middle is computed but zeroed.
- Residual memory for the backward pass is about 512 bytes per token per head (the saved 128x128 inverses) plus one `N x N` state per 128 tokens per head.
- Solve blocks are fixed at 16 tokens on purpose; do not look for a "bigger chunk = faster" switch, it is numerically unsafe.

**Attention**
- Head size multiple of 8 (lowered and tested for 64, 128, 256); `block` a multiple of 128 (128 or 256 recommended).
- Self-attention only (`Tq == Tk`), same number of heads for q/k/v (no GQA/MQA), softmax scale fixed to `1/sqrt(hd)`, no dropout, no additive bias, no sliding window (document mask only), no KV-cache decode path.
- This is the Transformer attention **kernel**, not an integration with the Hugging Face `transformers` library.

**General**
- JAX only (no PyTorch / PyTorch-XLA).
- The `'pallas'` backend off-TPU runs in Pallas interpret mode: correct but very slow, meant for tests only.

## Compatibility

| | status |
|---|---|
| Python | 3.10+ |
| JAX | validated with 0.12.0.1 (CPU tests, TPU lowering) and, by the author on real TPU runs, the pair `jax 0.11.2` + `libtpu 0.0.48`. Older JAX versions are untested. A `libtpu` older than your JAX's Mosaic compiler fails at load with *"Unsupported version"*: upgrade both together. |
| TPU hardware | trained and timed on **TPU v5e** (Colab / Kaggle runtimes). Other TPU generations are expected to work but were not tested. |
| CPU / GPU | XLA fallback (any backend JAX supports) |
| Frameworks | any JAX code: Flax, Equinox, Haiku, Optax, `jit`, `grad`, `jax.checkpoint` (`vmap` is untested) |
| OS (Rust wheels) | Linux x86_64 built and tested; macOS / Windows / aarch64 wheels are produced by the included CI workflow |

## What was tested

- `cargo test`: Rust unit tests (block tables, document ids, hand-computed RWKV-7 steps, attention masks).
- `pytest tests`: Rust vs NumPy fallback; Pallas kernels (interpret mode) vs the token-by-token recurrence and the dense float32 attention, forward **and** gradients, including identical-key (repeated-text) inputs, non-multiple lengths, padding, bf16; Rust CPU kernels vs JAX; and **Mosaic lowering for TPU** of every kernel (forward and backward) via `jax.export`.
- Not covered by this repository's CI: execution on a physical TPU. TPU numerics and the speed figure come from the author's training runs; run `benchmarks/bench.py` and `autotune` on your own TPU before relying on them.

## Layout

```
python/verokernel/   ops.py (public API), _rwkv7_kernel.py, _flash_kernel.py, _common.py, reference.py, host.py, autotune.py, _fallback.py
src/                 lib.rs (PyO3 bindings), core.rs (pure Rust: host helpers + CPU kernels, unit-tested)
tests/  examples/  benchmarks/  .github/workflows/build.yml (unpack + test + wheels + release)
```

## License

MIT (see `LICENSE`). RWKV-7 is the architecture of Peng et al.; flash attention follows Dao et al.; this repository is an independent implementation.
