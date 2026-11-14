//! VeroKernel native extension: Python bindings over `core` (host-side helpers and CPU reference kernels).
pub mod core;

#[cfg(feature = "python")]
mod py {
    use super::core;
    use numpy::{PyArray1, PyReadonlyArray1};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    fn as_s<'a, T: numpy::Element>(a: &'a PyReadonlyArray1<'a, T>) -> PyResult<&'a [T]> {
        a.as_slice().map_err(|_| PyValueError::new_err("array must be C-contiguous"))
    }

    /// block_doc_ranges(doc[i32; B*T], valid[u8; B*T], b, t, blk) -> (lo, hi)  (flat int32 arrays of length B*T/blk)
    #[pyfunction]
    fn block_doc_ranges<'py>(py: Python<'py>, doc: PyReadonlyArray1<'py, i32>, valid: PyReadonlyArray1<'py, u8>, b: usize, t: usize, blk: usize)
        -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>)> {
        let (d, v) = (as_s(&doc)?, as_s(&valid)?);
        if blk == 0 || t % blk != 0 || d.len() != b * t || v.len() != b * t {
            return Err(PyValueError::new_err("block_doc_ranges: need doc/valid of B*T elements and T % block == 0"));
        }
        let (lo, hi) = py.allow_threads(|| core::block_doc_ranges(d, v, b, t, blk));
        Ok((PyArray1::from_vec_bound(py, lo), PyArray1::from_vec_bound(py, hi)))
    }

    #[pyfunction]
    fn valid_length<'py>(py: Python<'py>, doc: PyReadonlyArray1<'py, i32>, b: usize, t: usize, pad: i32) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let d = as_s(&doc)?;
        if d.len() != b * t { return Err(PyValueError::new_err("valid_length: doc must have B*T elements")); }
        Ok(PyArray1::from_vec_bound(py, core::valid_length(d, b, t, pad)))
    }

    #[pyfunction]
    fn doc_ids_from_lengths<'py>(py: Python<'py>, lengths: PyReadonlyArray1<'py, i32>, b: usize, n: usize, t: usize, pad: i32) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let l = as_s(&lengths)?;
        if l.len() != b * n { return Err(PyValueError::new_err("doc_ids_from_lengths: lengths must have B*n elements")); }
        Ok(PyArray1::from_vec_bound(py, core::doc_ids_from_lengths(l, b, n, t, pad)))
    }

    #[pyfunction]
    fn doc_ids_from_starts<'py>(py: Python<'py>, starts: PyReadonlyArray1<'py, u8>, b: usize, t: usize) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let s = as_s(&starts)?;
        if s.len() != b * t { return Err(PyValueError::new_err("doc_ids_from_starts: starts must have B*T elements")); }
        Ok(PyArray1::from_vec_bound(py, core::doc_ids_from_starts(s, b, t)))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn wkv7_forward<'py>(py: Python<'py>, r: PyReadonlyArray1<'py, f32>, lw: PyReadonlyArray1<'py, f32>, k: PyReadonlyArray1<'py, f32>, v: PyReadonlyArray1<'py, f32>,
        kk: PyReadonlyArray1<'py, f32>, a: PyReadonlyArray1<'py, f32>, doc: PyReadonlyArray1<'py, i32>, b: usize, t: usize, h: usize, n: usize)
        -> PyResult<Bound<'py, PyArray1<f32>>> {
        let total = b * t * h * n;
        let xs = [as_s(&r)?, as_s(&lw)?, as_s(&k)?, as_s(&v)?, as_s(&kk)?, as_s(&a)?];
        let d = as_s(&doc)?;
        if xs.iter().any(|x| x.len() != total) || d.len() != b * t {
            return Err(PyValueError::new_err("wkv7_forward: inputs must have B*T*H*N elements and doc B*T"));
        }
        let y = py.allow_threads(|| core::wkv7_forward(xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], d, b, t, h, n));
        Ok(PyArray1::from_vec_bound(py, y))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn attention_forward<'py>(py: Python<'py>, q: PyReadonlyArray1<'py, f32>, k: PyReadonlyArray1<'py, f32>, v: PyReadonlyArray1<'py, f32>,
        doc: PyReadonlyArray1<'py, i32>, valid: PyReadonlyArray1<'py, u8>, b: usize, h: usize, t: usize, hd: usize, causal: bool)
        -> PyResult<(Bound<'py, PyArray1<f32>>, f32)> {
        let total = b * h * t * hd;
        let (qs, ks, vs, d, va) = (as_s(&q)?, as_s(&k)?, as_s(&v)?, as_s(&doc)?, as_s(&valid)?);
        if qs.len() != total || ks.len() != total || vs.len() != total || d.len() != b * t || va.len() != b * t {
            return Err(PyValueError::new_err("attention_forward: q,k,v must have B*H*T*hd elements, doc/valid B*T"));
        }
        let (o, mx) = py.allow_threads(|| core::attention_forward(qs, ks, vs, d, va, b, h, t, hd, causal));
        Ok((PyArray1::from_vec_bound(py, o), mx))
    }

    #[pyfunction]
    fn num_threads() -> usize { core::num_threads() }

    #[pymodule]
    pub fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(block_doc_ranges, m)?)?;
        m.add_function(wrap_pyfunction!(valid_length, m)?)?;
        m.add_function(wrap_pyfunction!(doc_ids_from_lengths, m)?)?;
        m.add_function(wrap_pyfunction!(doc_ids_from_starts, m)?)?;
        m.add_function(wrap_pyfunction!(wkv7_forward, m)?)?;
        m.add_function(wrap_pyfunction!(attention_forward, m)?)?;
        m.add_function(wrap_pyfunction!(num_threads, m)?)?;
        m.add("PAD_DOC", core::PAD_DOC)?;
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use py::_native;
