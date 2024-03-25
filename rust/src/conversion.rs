//! Conversion functions that can be used to convert pyQuil types to [`quil_rs`] types.
//!
//! This module contains two kinds of functions:
//! - `py_to_${quil_rs::t}` functions convert a #[pyo3::PyAny] into some quil-rs type `t`. These methods are
//!    compatible with the #[pyo3(from_py_with="")] field attribute for function arguments.
//! - `${quil_rs::t}_from_{k}` functions return some quil-rs type `t` from some other type `k`.
//!
//! Both types of functions will raise an error if the input values cannot be used to construct a
//! valid quil-rs type.
//!
//! This module should generally be used to convert pyo3 base types or Rust atomics to a quil-rs
//! type that has no equivalent class in PyQuil already. If a PyQuil class exists for that type
//! prefer to use it and [`std::convert::From`] to convert to a quil-rs type. For example, pyQuil
//! expresses offsets as (u64, String).
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use quil_rs::quil::Quil;

/// Converts a pyQuil's `list[tuple[int, str]]` to a quil-rs vector of offsets.
pub(crate) fn py_to_offsets(
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<quil_rs::instruction::Offset>> {
    let offsets: Option<Vec<(u64, String)>> = value.extract()?;
    offsets
        .unwrap_or_default()
        .into_iter()
        .map(offset_from_tuple)
        .collect()
}

/// Converts a pyQuil's Tuple[int, str] to a quil-rs offset.
pub(crate) fn py_to_offset(value: &Bound<'_, PyAny>) -> PyResult<quil_rs::instruction::Offset> {
    let offset_tuple: (u64, String) = value.extract()?;
    offset_from_tuple(offset_tuple)
}

pub(crate) fn offset_from_tuple(
    (length, memory_type): (u64, String),
) -> PyResult<quil_rs::instruction::Offset> {
    Ok(quil_rs::instruction::Offset::new(
        length,
        scalar_type_from_string(memory_type)?,
    ))
}

pub(crate) fn tuple_from_offset(offset: quil_rs::instruction::Offset) -> PyResult<(u64, String)> {
    Ok((offset.offset, string_from_scalar_type(offset.data_type)?))
}

pub(crate) fn optional_tuples_from_offsets(
    offsets: Vec<quil_rs::instruction::Offset>,
) -> PyResult<Option<Vec<(u64, String)>>> {
    let tuples: Vec<(u64, String)> = offsets
        .into_iter()
        .map(tuple_from_offset)
        .collect::<PyResult<_>>()?;
    if tuples.is_empty() {
        Ok(None)
    } else {
        Ok(Some(tuples))
    }
}

/// Converts a pyQuil memory type string to a quil-rs ScalarType
pub(crate) fn py_to_scalar_type(
    value: &Bound<'_, PyAny>,
) -> PyResult<quil_rs::instruction::ScalarType> {
    let memory_type: String = value.extract()?;
    scalar_type_from_string(memory_type)
}

pub(crate) fn scalar_type_from_string(s: String) -> PyResult<quil_rs::instruction::ScalarType> {
    match s.to_uppercase().as_str() {
        "BIT" => Ok(quil_rs::instruction::ScalarType::Bit),
        "INT" => Ok(quil_rs::instruction::ScalarType::Integer),
        "REAL" => Ok(quil_rs::instruction::ScalarType::Real),
        "OCTET" => Ok(quil_rs::instruction::ScalarType::Octet),
        _ => Err(PyValueError::new_err(
            "{} is not a valid memory type. Must be BIT, INT, REAL, or OCTET.",
        )),
    }
}

pub(crate) fn string_from_scalar_type(
    scalar_type: quil_rs::instruction::ScalarType,
) -> PyResult<String> {
    scalar_type.to_quil().map_err(|e| {
        PyRuntimeError::new_err(format!("Could not convert scalar type to Quil string: {e}"))
    })
}
