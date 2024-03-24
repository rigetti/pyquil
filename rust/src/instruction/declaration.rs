use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::conversion::{optional_tuples_from_offsets, string_from_scalar_type};
use crate::instruction::Instruction;
use crate::{conversion, extract_instruction_as, extract_instruction_as_mut, impl_from_quil_rs};

#[pyclass(extends=Instruction)]
#[derive(Debug, Clone)]
pub struct Declare {}
impl_from_quil_rs!(Declare, quil_rs::instruction::Declaration, Declaration);

#[pymethods]
impl Declare {
    #[new]
    #[pyo3(signature=(name, memory_type, memory_size=1, shared_region=None, offsets=Vec::new()))]
    pub fn new(
        name: String,
        #[pyo3(from_py_with = "conversion::py_to_scalar_type")]
        memory_type: quil_rs::instruction::ScalarType,
        memory_size: u64,
        shared_region: Option<String>,
        #[pyo3(from_py_with = "conversion::py_to_offsets")] offsets: Vec<
            quil_rs::instruction::Offset,
        >,
    ) -> PyResult<(Self, Instruction)> {
        let sharing = shared_region.map(|name| quil_rs::instruction::Sharing::new(name, offsets));

        Ok((
            Self {},
            Instruction {
                inner: quil_rs::instruction::Instruction::Declaration(
                    quil_rs::instruction::Declaration::new(
                        name,
                        quil_rs::instruction::Vector::new(memory_type, memory_size),
                        sharing,
                    ),
                ),
            },
        ))
    }

    #[getter]
    fn memory_type(self_: PyRef<'_, Self>) -> PyResult<String> {
        let instruction = self_.into_super();
        let declaration = extract_instruction_as!(instruction, Declaration)?;
        string_from_scalar_type(declaration.size.data_type)
    }

    #[setter]
    fn set_memory_type(self_: PyRefMut<'_, Self>, memory_type: Bound<'_, PyAny>) -> PyResult<()> {
        let mut instruction = self_.into_super();
        let memory_type = conversion::py_to_scalar_type(&memory_type)?;
        let declaration = extract_instruction_as_mut!(instruction, Declaration)?;
        declaration.size.data_type = memory_type;
        Ok(())
    }

    #[getter]
    fn memory_size(self_: PyRef<'_, Self>) -> PyResult<u64> {
        let instruction = self_.into_super();
        let declaration = extract_instruction_as!(instruction, Declaration)?;
        Ok(declaration.size.length)
    }

    #[setter]
    fn set_memory_size(self_: PyRefMut<'_, Self>, memory_size: u64) -> PyResult<()> {
        let mut instruction = self_.into_super();
        let declaration = extract_instruction_as_mut!(instruction, Declaration)?;
        declaration.size.length = memory_size;
        Ok(())
    }

    #[getter]
    fn shared_region(self_: PyRef<'_, Self>) -> PyResult<Option<String>> {
        let instruction = self_.into_super();
        let declaration = extract_instruction_as!(instruction, Declaration)?;
        Ok(declaration.sharing.clone().map(|sharing| sharing.name))
    }

    #[setter]
    fn set_shared_region(self_: PyRefMut<'_, Self>, shared_region: Option<String>) -> PyResult<()> {
        let mut instruction = self_.into_super();
        let declaration = extract_instruction_as_mut!(instruction, Declaration)?;
        declaration.sharing = shared_region.map(|region_name| {
            quil_rs::instruction::Sharing::new(
                region_name,
                declaration
                    .sharing
                    .take()
                    .map(|sharing| sharing.offsets)
                    .unwrap_or_default(),
            )
        });
        Ok(())
    }

    #[getter]
    fn offsets(self_: PyRef<'_, Self>) -> PyResult<Option<Vec<(u64, String)>>> {
        let instruction = self_.into_super();
        let declaration = extract_instruction_as!(instruction, Declaration)?;
        Ok(match &declaration.sharing {
            None => None,
            Some(sharing) => optional_tuples_from_offsets(sharing.offsets.clone())?,
        })
    }

    #[setter]
    fn set_offsets(self_: PyRefMut<'_, Self>, offsets: Bound<'_, PyAny>) -> PyResult<()> {
        let mut instruction = self_.into_super();
        let offsets = conversion::py_to_offsets(&offsets)?;
        let declaration = extract_instruction_as_mut!(instruction, Declaration)?;
        match declaration.sharing {
            None => {}
            Some(ref mut sharing) => sharing.offsets = offsets,
        }
        Ok(())
    }

    fn asdict<'a>(self_: PyRef<'a, Self>, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        let instruction = self_.into_super();
        let declaration: &quil_rs::instruction::Declaration =
            extract_instruction_as!(instruction, Declaration)?;
        let dict = PyDict::new_bound(py);
        dict.set_item("name", declaration.name.clone())?;
        dict.set_item(
            "memory_type",
            string_from_scalar_type(declaration.size.data_type)?,
        )?;
        dict.set_item("memory_size", declaration.size.length)?;
        dict.set_item(
            "shared_region",
            declaration.sharing.clone().map(|sharing| sharing.name),
        )?;
        dict.set_item(
            "offsets",
            match &declaration.sharing {
                None => None,
                Some(sharing) => optional_tuples_from_offsets(sharing.offsets.clone())?,
            },
        )?;

        Ok(dict)
    }
}
