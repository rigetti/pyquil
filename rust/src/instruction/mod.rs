use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyBytes,
};

use std::{collections::HashMap, str::FromStr};

use quil_rs::quil::Quil;

mod declaration;
mod gate;
mod quilt;

pub use declaration::Declare;
pub use gate::*;
pub use quilt::*;

use crate::impl_eq;

pub fn init_module(_py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new_bound(_py, "instruction")?;
    m.add_class::<Instruction>()?;
    m.add_class::<DefCalibration>()?;
    m.add_class::<DefMeasureCalibration>()?;
    m.add_class::<GateModifier>()?;
    Ok(m)
}

#[pyclass(subclass)]
#[derive(Clone, Debug, PartialEq)]
pub struct Instruction {
    inner: quil_rs::instruction::Instruction,
}
impl_eq!(Instruction);

impl From<Instruction> for quil_rs::instruction::Instruction {
    fn from(instruction: Instruction) -> Self {
        instruction.inner
    }
}

impl From<quil_rs::instruction::Instruction> for Instruction {
    fn from(instruction: quil_rs::instruction::Instruction) -> Self {
        Self { inner: instruction }
    }
}

#[pymethods]
impl Instruction {
    fn out(&self) -> PyResult<String> {
        self.inner
            .to_quil()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn __str__(&self) -> String {
        self.inner.to_quil_or_debug()
    }

    pub fn __repr__(&self) -> String {
        self.inner.to_quil_or_debug()
    }

    pub fn __deepcopy__(self_: PyRef<'_, Self>, _memo: &pyo3::types::PyDict) -> Self {
        let mut instruction = Self {
            inner: self_.inner.clone(),
        };

        // QubitPlaceholders are implemented with Arc and identified as unique by their pointer
        // address. Since cloning an Arc just copies the pointer address, we have to create new
        // placeholders for the copy, otherwise resolving a placeholder in the copy would also
        // resolve them in the original (or vice-versa).
        let mut placeholders: HashMap<
            quil_rs::instruction::QubitPlaceholder,
            quil_rs::instruction::QubitPlaceholder,
        > = HashMap::new();

        for qubit in instruction.inner.get_qubits_mut() {
            match qubit {
                quil_rs::instruction::Qubit::Fixed(_)
                | quil_rs::instruction::Qubit::Variable(_) => *qubit = qubit.clone(),
                quil_rs::instruction::Qubit::Placeholder(placeholder) => {
                    *qubit = quil_rs::instruction::Qubit::Placeholder(
                        placeholders.entry(placeholder.clone()).or_default().clone(),
                    )
                }
            }
        }

        instruction
    }

    pub fn __copy__(&self) -> Self {
        self.clone()
    }

    // This will raise an error if the program contains any unresolved
    // placeholders. This is because they can't be converted to valid quil,
    // nor can they be serialized and deserialized in a consistent
    // way.
    pub fn __getstate__<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyBytes>> {
        Ok(PyBytes::new_bound(
            py,
            self.inner
                .to_quil()
                .map_err(|e| {
                    PyValueError::new_err(format!("Could not serialize instruction: {}", e,))
                })?
                .as_bytes(),
        ))
    }

    pub fn __setstate__<'a>(
        &mut self,
        _py: Python<'a>,
        state: &Bound<'a, PyBytes>,
    ) -> PyResult<()> {
        let program_str = std::str::from_utf8(state.as_bytes()).map_err(|e| {
            PyValueError::new_err(format!("Could not deserialize non-utf-8 string: {}", e))
        })?;
        let instructions = quil_rs::program::Program::from_str(program_str)
            .map_err(|e| PyRuntimeError::new_err(format!("Could not deserialize {}", e)))?
            .into_instructions();

        if instructions.len() != 1 {
            return Err(PyRuntimeError::new_err(format!(
                "Expected to deserialize a single instruction, got {}: {:?}",
                instructions.len(),
                instructions
            )));
        }

        *self = Instruction {
            inner: instructions
                .into_iter()
                .next()
                .expect("instructions has exactly one instruction."),
        };

        Ok(())
    }
}

impl Instruction {
    pub(crate) fn from_quil_rs(instruction: quil_rs::instruction::Instruction) -> Self {
        Self { inner: instruction }
    }
}

#[macro_export]
macro_rules! impl_from_quil_rs(
        ($name: ident, $rs_instruction: path, $variant_name: ident) => {
        impl $name {
            pub(crate) fn from_quil_rs(py: Python<'_>, rs_instruction: $rs_instruction) -> pyo3::PyResult<pyo3::PyObject> {
                let instruction = Instruction {
                    inner: quil_rs::instruction::Instruction::$variant_name(rs_instruction)
                };

                // This roundabout way of returning an Instruction subclass is necessary because
                // a pyclass that extends another does not implement IntoPy, which makes it
                // impossible to return from a pymethod. This is currently the only way to return
                // a child class.
                // https://github.com/PyO3/pyo3/issues/1836
                use pyo3::PyTypeInfo;
                use pyo3::pyclass_init::PyObjectInit;

                // Adding a subclass of Self isn't strictly necessary, but it ensures that Self is
                // truly a class that extends Instruction. If it's not, then the compiler will
                // raise an error.
                let initializer = pyo3::pyclass_init::PyClassInitializer::from(instruction).add_subclass::<Self>(Self {});

                // SAFETY: `into_new_object` requires that the provided subtype be a valid
                // pointer to a type object of the initializers T. We derive the type
                // object from `Self`, which is also used to construct the initializer.
                unsafe {
                    initializer
                        .into_new_object(py, Self::type_object_raw(py))
                        .map(|ptr| PyObject::from_owned_ptr(py, ptr))
                }
            }
        }
    }
);

/// Attempts to return a reference to the inner quil-rs instruction of an [`Instruction`]. Returns
/// a result containing the reference if extraction was successful, or an error if the desired
/// type didn't match the contents of the instruction.
///
/// * $py_ref must be of type [`PyRef<'_, Instruction>`].
/// * $dst_type is the name of the variant as it appears on the
///     [`quil_rs::instruction::Instruction`] enum.
#[macro_export]
macro_rules! extract_instruction_as(
    ($py_ref: ident, $dst_type: ident) => {
        {
            if let quil_rs::instruction::Instruction::$dst_type(__inner) = &$py_ref.inner {
                Ok(__inner)
            } else {
                Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Expected {} instruction, got: {:?}", stringify!($dst_type), $py_ref.inner)))
            }
        }
    }
);

/// Attempts to return a mutable reference to the inner quil-rs instruction of an [`Instruction`].
/// Returns a result containing the reference if extraction was successful, or an error if the
/// desired type didn't match the contents of the instruction.
///
/// * $py_ref must be of type [`PyRefMut<'_, Instruction>`].
/// * $dst_type is the name of the variant as it appears on the
///     [`quil_rs::instruction::Instruction`] enum.
#[macro_export]
macro_rules! extract_instruction_as_mut(
    ($py_ref_mut: ident, $dst_type: ident) => {
        {
            if let quil_rs::instruction::Instruction::$dst_type(__inner) = &mut $py_ref_mut.inner {
                Ok(__inner)
            } else {
                Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Expected {} instruction, got: {:?}", stringify!($dst_type), $py_ref_mut.inner)))
            }
        }
    }
);
