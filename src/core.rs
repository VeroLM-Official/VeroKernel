//! Pure-Rust implementations (no Python types): host-side data preparation and CPU reference kernels.
//!
//! Layouts are C-contiguous: RWKV-7 tensors are (B,T,H,N), attention tensors are (B,H,T,hd), document ids are (B,T).

use std::sync::atomic::{AtomicUsize, Ordering};

/// Document id that marks padding rows (identical to `verokernel.PAD_DOC`).
pub const PAD_DOC: i32 = 1 << 28;
/// "No valid row in this block" sentinel for the lower bound of a block's document range.
pub const BIG_I: i32 = 1 << 30;

pub fn num_threads() -> usize {
    if let Ok(v) = std::env::var("VEROKERNEL_THREADS") {
        if let Ok(n) = v.parse::<usize>() {
            if n >= 1 {
                return n;
            }
        }
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Run `f(task)` for every task in 0..n_tasks on a small thread pool and return the results in task order.
fn parallel_map<R: Send, F: Fn(usize) -> R + Sync>(n_tasks: usize, f: F) -> Vec<R> {
    let workers = num_threads().min(n_tasks.max(1));
    if workers <= 1 {
        return (0..n_tasks).map(&f).collect();
    }
    let next = AtomicUsize::new(0);
    let mut parts: Vec<Vec<(usize, R)>> = Vec::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                s.spawn(|| {
                    let mut local = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n_tasks {
                            break;
                        }
                        local.push((i, f(i)));
                    }
                    local
                })
            })
            .collect();
        for h in handles {
            parts.push(h.join().expect("worker thread panicked"));
        }
    });
    let mut all: Vec<(usize, R)> = parts.into_iter().flatten().collect();
    all.sort_by_key(|x| x.0);
    all.into_iter().map(|x| x.1).collect()
}

// ------------------------------------------------------------------------------------------------ host-side data preparation

/// For every block of `blk` rows: smallest / largest document id among its VALID rows (empty block: lo = BIG_I, hi = -1).
/// `doc`, `valid`: (B,T) with T % blk == 0. Returns two flat vectors of length B*T/blk. Same output as the scalar-prefetch tables of the flash kernel.
pub fn block_doc_ranges(doc: &[i32], valid: &[u8], b: usize, t: usize, blk: usize) -> (Vec<i32>, Vec<i32>) {
    assert!(blk > 0 && t % blk == 0, "T must be a multiple of the block size");
    assert_eq!(doc.len(), b * t);
    assert_eq!(valid.len(), b * t);
    let n = t / blk;
    let mut lo = vec![BIG_I; b * n];
    let mut hi = vec![-1i32; b * n];
    for bi in 0..b {
        for ni in 0..n {
            let base = bi * t + ni * blk;
            let (mut l, mut h) = (BIG_I, -1i32);
            for i in 0..blk {
                if valid[base + i] != 0 {
                    let d = doc[base + i];
                    if d < l {
                        l = d;
                    }
                    if d > h {
                        h = d;
                    }
                }
            }
            lo[bi * n + ni] = l;
            hi[bi * n + ni] = h;
        }
    }
    (lo, hi)
}

/// Number of leading rows that contain real data: (index of the last row with doc < pad) + 1, or 0. Returns (B,) values.
pub fn valid_length(doc: &[i32], b: usize, t: usize, pad: i32) -> Vec<i32> {
    assert_eq!(doc.len(), b * t);
    (0..b)
        .map(|bi| {
            let row = &doc[bi * t..(bi + 1) * t];
            row.iter().rposition(|&d| d < pad).map(|p| p as i32 + 1).unwrap_or(0)
        })
        .collect()
}

/// Document ids from document lengths. `lengths` is (B,n); entries <= 0 are skipped. Rows beyond the last document become `pad`.
/// Documents that do not fit in T rows are truncated. Returns (B,T).
pub fn doc_ids_from_lengths(lengths: &[i32], b: usize, n: usize, t: usize, pad: i32) -> Vec<i32> {
    assert_eq!(lengths.len(), b * n);
    let mut out = vec![pad; b * t];
    for bi in 0..b {
        let mut pos = 0usize;
        let mut id = 0i32;
        for j in 0..n {
            let len = lengths[bi * n + j];
            if len <= 0 {
                continue;
            }
            let end = (pos + len as usize).min(t);
            for p in pos..end {
                out[bi * t + p] = id;
            }
            pos = end;
            id += 1;
            if pos >= t {
                break;
            }
        }
    }
    out
}

/// Document ids from a "document starts here" mask (B,T): id[t] = number of starts in 1..=t (position 0 always opens document 0).
pub fn doc_ids_from_starts(starts: &[u8], b: usize, t: usize) -> Vec<i32> {
    assert_eq!(starts.len(), b * t);
    let mut out = vec![0i32; b * t];
    for bi in 0..b {
        let mut id = 0i32;
        for ti in 0..t {
            if ti > 0 && starts[bi * t + ti] != 0 {
                id += 1;
            }
            out[bi * t + ti] = id;
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------ CPU reference: RWKV-7

/// Sequential RWKV-7 recurrence (the definition), float32. All tensors (B,T,H,N); `doc` (B,T).
/// The state of a head is reset whenever the document id changes; rows with doc >= PAD_DOC produce zeros (as the Pallas kernel does).
/// state S[i][j]: S <- S*exp(lw)[j] - (S kk)[i] * (kk*a)[j] + v[i] * k[j];  y[i] = sum_j S[i][j] r[j]
#[allow(clippy::too_many_arguments)]
pub fn wkv7_forward(
    r: &[f32], lw: &[f32], k: &[f32], v: &[f32], kk: &[f32], a: &[f32], doc: &[i32],
    b: usize, t: usize, h: usize, n: usize,
) -> Vec<f32> {
    let total = b * t * h * n;
    for x in [r, lw, k, v, kk, a] {
        assert_eq!(x.len(), total, "all RWKV-7 inputs must have B*T*H*N elements");
    }
    assert_eq!(doc.len(), b * t);
    let parts = parallel_map(b * h, |task| {
        let (bi, hi) = (task / h, task % h);
        let mut s = vec![0f32; n * n];
        let mut out = vec![0f32; t * n];
        let mut skk = vec![0f32; n];
        let mut decay = vec![0f32; n];
        let mut bvec = vec![0f32; n];
        for ti in 0..t {
            let d = doc[bi * t + ti];
            if ti > 0 && d != doc[bi * t + ti - 1] {
                s.iter_mut().for_each(|x| *x = 0.0);
            }
            if d >= PAD_DOC {
                continue; // padding row: state stays zero (reset above), output zero
            }
            let off = ((bi * t + ti) * h + hi) * n;
            let (rr, ll, kt, vt, kkt, at) = (&r[off..off + n], &lw[off..off + n], &k[off..off + n], &v[off..off + n], &kk[off..off + n], &a[off..off + n]);
            for j in 0..n {
                decay[j] = ll[j].exp();
                bvec[j] = kkt[j] * at[j];
            }
            for i in 0..n {
                let row = &s[i * n..(i + 1) * n];
                skk[i] = row.iter().zip(kkt).map(|(x, y)| x * y).sum();
            }
            for i in 0..n {
                let (c, vi) = (skk[i], vt[i]);
                let row = &mut s[i * n..(i + 1) * n];
                let mut acc = 0f32;
                for j in 0..n {
                    let nv = row[j] * decay[j] - c * bvec[j] + vi * kt[j];
                    row[j] = nv;
                    acc += nv * rr[j];
                }
                out[ti * n + i] = acc;
            }
        }
        out
    });
    let mut y = vec![0f32; total];
    for (task, buf) in parts.into_iter().enumerate() {
        let (bi, hi) = (task / h, task % h);
        for ti in 0..t {
            let off = ((bi * t + ti) * h + hi) * n;
            y[off..off + n].copy_from_slice(&buf[ti * n..(ti + 1) * n]);
        }
    }
    y
}

// ------------------------------------------------------------------------------------------------ CPU reference: attention

/// Document-masked attention with an online (flash-style) softmax, float32. q,k,v (B,H,T,hd); `doc` (B,T); `valid` (B,T) as 0/1.
/// A query attends to keys of its own document that are valid (and, if `causal`, not later than itself). Queries without any such key output zeros.
/// Returns (o, max_logit) where max_logit is the largest attended score (floored at -1e4, like the kernel).
#[allow(clippy::too_many_arguments)]
pub fn attention_forward(
    q: &[f32], k: &[f32], v: &[f32], doc: &[i32], valid: &[u8],
    b: usize, h: usize, t: usize, hd: usize, causal: bool,
) -> (Vec<f32>, f32) {
    let total = b * h * t * hd;
    assert!(q.len() == total && k.len() == total && v.len() == total, "q, k, v must have B*H*T*hd elements");
    assert_eq!(doc.len(), b * t);
    assert_eq!(valid.len(), b * t);
    const ROWS: usize = 32;
    let chunks = (t + ROWS - 1) / ROWS;
    let scale = 1.0 / (hd as f32).sqrt();
    let parts = parallel_map(b * h * chunks, |task| {
        let (bh, c) = (task / chunks, task % chunks);
        let bi = bh / h;
        let (r0, r1) = (c * ROWS, ((c + 1) * ROWS).min(t));
        let base = bh * t * hd;
        let mut out = vec![0f32; (r1 - r0) * hd];
        let mut mx = -1e4f32;
        let mut acc = vec![0f32; hd];
        for i in r0..r1 {
            let qi = &q[base + i * hd..base + (i + 1) * hd];
            let di = doc[bi * t + i];
            let (mut m, mut l) = (f32::NEG_INFINITY, 0f32);
            acc.iter_mut().for_each(|x| *x = 0.0);
            let jmax = if causal { i + 1 } else { t };
            for j in 0..jmax {
                if valid[bi * t + j] == 0 || doc[bi * t + j] != di {
                    continue;
                }
                let kj = &k[base + j * hd..base + (j + 1) * hd];
                let s: f32 = qi.iter().zip(kj).map(|(x, y)| x * y).sum::<f32>() * scale;
                if s > mx {
                    mx = s;
                }
                let m_new = if s > m { s } else { m };
                let alpha = if m == f32::NEG_INFINITY { 0.0 } else { (m - m_new).exp() };
                let p = (s - m_new).exp();
                l = alpha * l + p;
                let vj = &v[base + j * hd..base + (j + 1) * hd];
                for d in 0..hd {
                    acc[d] = alpha * acc[d] + p * vj[d];
                }
                m = m_new;
            }
            if l > 0.0 {
                for d in 0..hd {
                    out[(i - r0) * hd + d] = acc[d] / l;
                }
            }
        }
        (out, mx)
    });
    let mut o = vec![0f32; total];
    let mut max_logit = -1e4f32;
    for (task, (buf, mx)) in parts.into_iter().enumerate() {
        let (bh, c) = (task / chunks, task % chunks);
        let r0 = c * ROWS;
        let base = bh * t * hd + r0 * hd;
        o[base..base + buf.len()].copy_from_slice(&buf);
        if mx > max_logit {
            max_logit = mx;
        }
    }
    (o, max_logit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_skip_invalid_rows() {
        let doc = [0, 0, 1, 1, 5, 7, 9, 9];
        let valid = [1, 1, 1, 0, 1, 1, 0, 0];
        let (lo, hi) = block_doc_ranges(&doc, &valid, 1, 8, 2);
        assert_eq!(lo, vec![0, 1, 5, BIG_I]);
        assert_eq!(hi, vec![0, 1, 7, -1]);
    }

    #[test]
    fn valid_length_and_ids() {
        let doc = [0, 0, 1, PAD_DOC, PAD_DOC, 0, 0, PAD_DOC];
        assert_eq!(valid_length(&doc, 2, 4, PAD_DOC), vec![3, 3]);
        assert_eq!(valid_length(&[PAD_DOC, PAD_DOC], 1, 2, PAD_DOC), vec![0]);
        let ids = doc_ids_from_lengths(&[2, 0, 3, 9], 1, 4, 8, PAD_DOC);
        assert_eq!(ids, vec![0, 0, 1, 1, 1, 2, 2, 2]);
        let ids = doc_ids_from_lengths(&[2, 3], 1, 2, 8, PAD_DOC);
        assert_eq!(ids, vec![0, 0, 1, 1, 1, PAD_DOC, PAD_DOC, PAD_DOC]);
        assert_eq!(doc_ids_from_starts(&[1, 0, 1, 0, 0, 1], 1, 6), vec![0, 0, 1, 1, 1, 2]);
    }

    #[test]
    fn wkv_single_step_matches_hand_computation() {
        // N=1, one token: S = 0 -> S = v*k, y = S*r
        let y = wkv7_forward(&[2.0], &[-0.1], &[3.0], &[5.0], &[1.0], &[0.5], &[0], 1, 1, 1, 1);
        assert!((y[0] - 5.0 * 3.0 * 2.0).abs() < 1e-6);
        // two tokens, same doc: S1 = 15, S2 = 15*e^-0.1 - (15*kk)*(kk*a) + v*k with kk=1,a=0.5
        let y = wkv7_forward(&[1.0, 1.0], &[-0.1, -0.1], &[3.0, 3.0], &[5.0, 5.0], &[1.0, 1.0], &[0.5, 0.5], &[0, 0], 1, 2, 1, 1);
        let s2 = 15.0 * (-0.1f32).exp() - 15.0 * 0.5 + 15.0;
        assert!((y[1] - s2).abs() < 1e-4);
        // document boundary resets the state
        let y = wkv7_forward(&[1.0, 1.0], &[-0.1, -0.1], &[3.0, 3.0], &[5.0, 5.0], &[1.0, 1.0], &[0.5, 0.5], &[0, 1], 1, 2, 1, 1);
        assert!((y[1] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn attention_uniform_and_masks() {
        // hd=1, q=k=0 -> uniform weights over the allowed keys
        let (b, h, t, hd) = (1, 1, 4, 1);
        let q = vec![0.0; 4];
        let k = vec![0.0; 4];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let (o, _) = attention_forward(&q, &k, &v, &[0, 0, 0, 0], &[1, 1, 1, 1], b, h, t, hd, true);
        assert!((o[3] - 2.5).abs() < 1e-6 && (o[0] - 1.0).abs() < 1e-6);
        let (o, _) = attention_forward(&q, &k, &v, &[0, 0, 1, 1], &[1, 1, 1, 1], b, h, t, hd, true);
        assert!((o[3] - 3.5).abs() < 1e-6);
        let (o, _) = attention_forward(&q, &k, &v, &[0, 0, 0, 0], &[1, 1, 1, 0], b, h, t, hd, false);
        assert!((o[0] - 2.0).abs() < 1e-6);
        let (o, _) = attention_forward(&q, &k, &v, &[0, 0, 0, 9], &[1, 1, 1, 0], b, h, t, hd, false);
        assert_eq!(o[3], 0.0); // padding query without valid keys in its document
    }
}
