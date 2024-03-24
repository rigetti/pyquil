use std::str::FromStr;

use indexmap::IndexMap;
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::instruction::{Declare, DefCalibration, DefMeasureCalibration, Instruction};

#[pyclass]
#[derive(Clone, Debug)]
pub struct Program {
    inner: quil_rs::Program,
    #[pyo3(get, set)]
    num_shots: u64,
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
}

#[pymodule]
pub fn program(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Program>()?;
    Ok(())
}
