use pyo3::prelude::*;

pub(crate) fn init_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new_bound(py, "expression")?;
    m.add_class::<Expression>()?;
    m.add_class::<MemoryReference>()?;
    Ok(m)
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct Expression {}

impl From<Expression> for quil_rs::expression::Expression {
    fn from(_: Expression) -> Self {
        todo!()
    }
}

impl From<quil_rs::expression::Expression> for Expression {
    fn from(_: quil_rs::expression::Expression) -> Self {
        todo!()
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct MemoryReference {}

impl std::fmt::Display for MemoryReference {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl From<MemoryReference> for quil_rs::instruction::MemoryReference {
    fn from(_: MemoryReference) -> Self {
        todo!()
    }
}

impl From<quil_rs::instruction::MemoryReference> for MemoryReference {
    fn from(_: quil_rs::instruction::MemoryReference) -> Self {
        todo!()
    }
}
