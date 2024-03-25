use pyo3::prelude::*;

mod qubit;

pub use qubit::*;

pub(crate) fn init_module(py: Python) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new_bound(py, "primitive")?;
    m.add_class::<Qubit>()?;
    Ok(m)
}
