#![deny(clippy::all)]

use pyo3::prelude::*;

mod conversion;
mod expression;
mod instruction;
mod primitive;
mod program;

// A standard complex number type for the crate to use.
// This is equivalent to numpy's complex128 type.
pub type Complex = num_complex::Complex<f64>;

#[pymodule]
#[pyo3(name = "_core")]
fn pyquil(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    initalize_submodule(py, "instruction", m, &instruction::init_module(py)?)?;
    initalize_submodule(py, "expression", m, &expression::init_module(py)?)?;
    initalize_submodule(py, "primitive", m, &primitive::init_module(py)?)?;
    initalize_submodule(py, "program", m, &program::init_module(py)?)?;
    Ok(())
}

fn initalize_submodule(
    py: Python<'_>,
    name: &str,
    parent_module: &Bound<'_, PyModule>,
    submodule: &Bound<'_, PyModule>,
) -> PyResult<()> {
    parent_module.add_submodule(submodule)?;
    let sys_modules = py.import_bound("sys")?.getattr("modules")?;
    let qualified_name = format!(
        "{}.{}",
        parent_module.getattr("__name__")?.extract::<String>()?,
        name
    );
    sys_modules.set_item(&qualified_name, submodule)?;
    submodule.setattr("__name__", &qualified_name)?;
    Ok(())
}

/// Implement the __hash__ dunder method for a pyclass. The pyclass must impl [`std::hash::Hash`].
#[macro_export]
macro_rules! impl_hash {
    ($name: ident) => {
        #[pymethods]
        impl $name {
            fn __hash__(&self) -> u64 {
                use std::hash::{DefaultHasher, Hasher};
                let mut state = DefaultHasher::new();
                self.hash(&mut state);
                state.finish()
            }
        }
    };
}

/// Implement the == and != operators for a pyclass by implementing the __richcmp__ dunder method.
/// The pyclass must implement [`std::cmp::PartialEq`].
#[macro_export]
macro_rules! impl_eq {
    ($name: ident) => {
        #[pymethods]
        impl $name {
            fn __richcmp__(
                &self,
                other: &Self,
                op: pyo3::class::basic::CompareOp,
                py: Python<'_>,
            ) -> PyObject {
                match op {
                    pyo3::class::basic::CompareOp::Eq => (self == other).into_py(py),
                    pyo3::class::basic::CompareOp::Ne => (self != other).into_py(py),
                    _ => py.NotImplemented(),
                }
            }
        }
    };
}

/// Implement pickling for a pyclass. The pyclass must implement [`serde::Serialize`] and
/// [`serde::Deserialize`].
#[macro_export]
macro_rules! impl_pickle_for_serialize {
    ($name: ident) => {
        impl $name {
            pub fn __getstate__<'a>(
                &self,
                py: pyo3::Python<'a>,
            ) -> PyResult<pyo3::Bound<'a, pyo3::types::PyBytes>> {
                Ok(pyo3::types::PyBytes::new_bound(
                    py,
                    &bincode::serialize(&self).map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "Could not serialize: {}",
                            e.to_string()
                        ))
                    })?,
                ))
            }

            pub fn __setstate__<'a>(
                &mut self,
                _py: pyo3::Python<'a>,
                state: pyo3::Bound<'a, pyo3::types::PyBytes>,
            ) -> PyResult<()> {
                *self = bincode::deserialize(state.as_bytes()).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Could not serialize: {}",
                        e.to_string()
                    ))
                })?;
                Ok(())
            }
        }
    };
}
