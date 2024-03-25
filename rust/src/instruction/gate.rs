use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub enum GateModifier {
    Controlled,
    Dagger,
    Forked,
}

impl From<GateModifier> for quil_rs::instruction::GateModifier {
    fn from(modifier: GateModifier) -> Self {
        match modifier {
            GateModifier::Controlled => quil_rs::instruction::GateModifier::Controlled,
            GateModifier::Dagger => quil_rs::instruction::GateModifier::Dagger,
            GateModifier::Forked => quil_rs::instruction::GateModifier::Forked,
        }
    }
}

impl From<quil_rs::instruction::GateModifier> for GateModifier {
    fn from(modifier: quil_rs::instruction::GateModifier) -> Self {
        match modifier {
            quil_rs::instruction::GateModifier::Controlled => GateModifier::Controlled,
            quil_rs::instruction::GateModifier::Dagger => GateModifier::Dagger,
            quil_rs::instruction::GateModifier::Forked => GateModifier::Forked,
        }
    }
}
