use std::{str::FromStr, sync::Arc};

use indexmap::IndexMap;
use pyo3::{exceptions::PyValueError, prelude::*};
use quil_rs::quil::Quil;

use crate::instruction::{Declare, DefCalibration, DefMeasureCalibration, Instruction};

pub fn init_module(_py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new_bound(_py, "program")?;
    m.add_class::<Program>()?;
    Ok(m)
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct Program {
    inner: quil_rs::Program,
    #[pyo3(get, set)]
    num_shots: u64,
}

#[pyclass]
pub struct ProgramIter {
    // inner: std::vec::IntoIter<Instruction>,
    inner: Box<dyn Iterator<Item = Instruction> + Send>,
}

#[pymethods]
impl ProgramIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Instruction> {
        slf.inner.next()
    }
}

#[derive(FromPyObject, Clone, Debug)]
pub enum InstructionDesignator {
    Instruction(Instruction),
    // RsInstruction(quil_rs::instruction::Instruction),
    Serialized(String),
    Program(Program),
    // RsProgram(quil_rs::Program),
    // Sequence(Vec<InstructionDesignator>),
    // Tuple
    // Generator
}

#[pymethods]
impl Program {
    #[new]
    #[pyo3(signature=(instructions = None, *, num_shots = None))]
    fn new(instructions: Option<InstructionDesignator>, num_shots: Option<u64>) -> PyResult<Self> {
        let num_shots = num_shots.unwrap_or(1);
        Ok(match instructions {
            None => Self {
                inner: quil_rs::Program::new(),
                num_shots,
            },
            Some(InstructionDesignator::Instruction(instruction)) => Self {
                inner: quil_rs::Program::from_instructions(vec![instruction.into()]),
                num_shots,
            },
            // Some(InstructionDesignator::RsInstruction(instruction)) => Self {
            //     inner: quil_rs::Program::from_instructions(vec![instruction]),
            //     num_shots,
            // },
            Some(InstructionDesignator::Serialized(program)) => Self {
                inner: quil_rs::Program::from_str(&program).map_err(|e| {
                    PyValueError::new_err(format!("Failed to parse Quil program: {e}"))
                })?,
                num_shots,
            },
            Some(InstructionDesignator::Program(program)) => program.clone(),
            // Some(InstructionDesignator::RsProgram(program)) => Self {
            //     inner: program.clone(),
            //     num_shots,
            // },
        })
    }

    #[getter]
    fn calibrations(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.inner
            .calibrations
            .calibrations()
            .iter()
            .cloned()
            .map(|c| DefCalibration::from_quil_rs(py, c))
            .collect::<PyResult<Vec<PyObject>>>()
    }

    #[getter]
    fn measure_calibrations(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.inner
            .calibrations
            .measure_calibrations()
            .iter()
            .cloned()
            .map(|c| DefMeasureCalibration::from_quil_rs(py, c))
            .collect::<PyResult<Vec<PyObject>>>()
    }

    fn declarations(&self, py: Python<'_>) -> PyResult<IndexMap<String, PyObject>> {
        self.iter_declarations()
            .map(|declaration| {
                Ok((
                    declaration.name.clone(),
                    Declare::from_quil_rs(py, declaration)?,
                ))
            })
            .collect()
    }

    #[getter]
    fn instructions(&self) -> Vec<Instruction> {
        // pyQuil defines this property as Declarations + quil_rs body instructions
        self.iter_declarations()
            .map(|declaration| {
                Instruction::from_quil_rs(quil_rs::instruction::Instruction::Declaration(
                    declaration,
                ))
            })
            .chain(
                self.inner
                    .body_instructions()
                    .cloned()
                    .map(Instruction::from_quil_rs),
            )
            .collect()
    }

    fn out(&self) -> PyResult<String> {
        self.inner
            .to_quil()
            .map_err(|e| PyValueError::new_err(format!("Could not serialize program to Quil: {e}")))
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<ProgramIter>> {
        let iter = ProgramIter {
            inner: Box::new(self.clone().iter_instructions()),
        };
        Py::new(py, iter)
    }
}

impl Program {
    fn iter_declarations(&self) -> impl Iterator<Item = quil_rs::instruction::Declaration> {
        self.inner
            .memory_regions
            .clone()
            .into_iter()
            .map(|(name, descriptor)| {
                quil_rs::instruction::Declaration::new(name, descriptor.size, descriptor.sharing)
            })
    }

    fn iter_instructions(self) -> impl Iterator<Item = Instruction> {
        self.iter_declarations()
            .map(|declaration| {
                Instruction::from_quil_rs(quil_rs::instruction::Instruction::Declaration(
                    declaration,
                ))
            })
            .chain(
                self.inner
                    .into_body_instructions()
                    .map(Instruction::from_quil_rs),
            )
    }
}

#[pymodule]
pub fn program(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Program>()?;
    Ok(())
}
