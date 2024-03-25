use std::hash::Hash;

use pyo3::prelude::*;

use crate::{impl_eq, impl_hash};

#[derive(FromPyObject, Clone)]
pub enum QubitDesignator {
    Fixed(Qubit),
    FixedInt(u64),
    FormalArgument(FormalArgument),
    Placeholder(QubitPlaceholder),
}

impl From<QubitDesignator> for quil_rs::instruction::Qubit {
    fn from(designator: QubitDesignator) -> Self {
        match designator {
            QubitDesignator::Fixed(qubit) => quil_rs::instruction::Qubit::Fixed(qubit.index),
            QubitDesignator::FixedInt(qubit) => quil_rs::instruction::Qubit::Fixed(qubit),
            QubitDesignator::FormalArgument(argument) => {
                quil_rs::instruction::Qubit::Variable(argument.0)
            }
            QubitDesignator::Placeholder(placeholder) => {
                quil_rs::instruction::Qubit::Placeholder(placeholder.0)
            }
        }
    }
}

impl From<quil_rs::instruction::Qubit> for QubitDesignator {
    fn from(value: quil_rs::instruction::Qubit) -> Self {
        match value {
            quil_rs::instruction::Qubit::Fixed(index) => QubitDesignator::Fixed(Qubit { index }),
            quil_rs::instruction::Qubit::Variable(argument) => {
                QubitDesignator::FormalArgument(FormalArgument(argument))
            }
            quil_rs::instruction::Qubit::Placeholder(placeholder) => {
                QubitDesignator::Placeholder(QubitPlaceholder(placeholder))
            }
        }
    }
}

impl ToPyObject for QubitDesignator {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        match self {
            QubitDesignator::Fixed(qubit) => qubit.clone().into_py(py),
            QubitDesignator::FixedInt(qubit) => qubit.to_object(py),
            QubitDesignator::FormalArgument(argument) => argument.clone().into_py(py),
            QubitDesignator::Placeholder(placeholder) => placeholder.clone().into_py(py),
        }
    }
}

#[pyclass]
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct QubitPlaceholder(quil_rs::instruction::QubitPlaceholder);

#[pyclass]
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct FormalArgument(String);

#[pyclass]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Qubit {
    #[pyo3(get, set)]
    index: u64,
}
impl_hash!(Qubit);
impl_eq!(Qubit);

#[pymethods]
impl Qubit {
    #[new]
    fn __new__(index: u64) -> Self {
        Qubit { index }
    }

    fn out(&self) -> String {
        self.index.to_string()
    }

    fn __str__(&self) -> String {
        self.out()
    }

    fn __repr__(&self) -> String {
        format!("<Qubit {}>", self.index)
    }
}
