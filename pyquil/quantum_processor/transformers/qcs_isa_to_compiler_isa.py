"""Transforms a QCS ``InstructionSetArchitecture`` into a ``CompilerISA``."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Optional, Union, cast

import numpy as np
from qcs_sdk.qpu.isa import Characteristic, InstructionSetArchitecture, Operation

from pyquil.external.rpcq import (
    CompilerISA,
    GateInfo,
    MeasureInfo,
    Supported1QGate,
    Supported2QGate,
    add_edge,
    add_qubit,
    get_edge,
    get_qubit,
    make_edge_id,
)


class QCSISAParseError(ValueError):
    """Signals an error when creating a ``CompilerISA`` due to the operators in the QCS ``InstructionSetArchitecture``.

    This may raise as a consequence of unsupported gates as well as missing nodes or edges.
    """

    pass


def qcs_isa_to_compiler_isa(isa: InstructionSetArchitecture) -> CompilerISA:
    """Transform a QCS ``InstructionSetArchitecture`` into a ``CompilerISA``."""
    device = CompilerISA()
    for node in isa.architecture.nodes:
        add_qubit(device, node.node_id)

    for edge in isa.architecture.edges:
        add_edge(device, edge.node_ids[0], edge.node_ids[1])

    qubit_operations_seen: defaultdict[int, set[str]] = defaultdict(set)
    edge_operations_seen: defaultdict[str, set[str]] = defaultdict(set)
    for operation in isa.instructions:
        for site in operation.sites:
            if operation.node_count == 1:
                if len(site.node_ids) != 1:
                    raise QCSISAParseError(
                        f"operation {operation.name} has node count 1, but " f"site has {len(site.node_ids)} node_ids"
                    )
                operation_qubit = get_qubit(device, site.node_ids[0])
                if operation_qubit is None:
                    raise QCSISAParseError(
                        f"operation {operation.name} has node {site.node_ids[0]} "
                        "but node not declared in architecture"
                    )

                if operation.name in qubit_operations_seen[operation_qubit.id]:
                    continue
                qubit_operations_seen[operation_qubit.id].add(operation.name)

                operation_qubit.gates.extend(
                    _transform_qubit_operation_to_gates(
                        operation.name,
                        operation_qubit.id,
                        site.characteristics,
                        isa.benchmarks,
                    )
                )

            elif operation.node_count == 2:
                if len(site.node_ids) != 2:
                    QCSISAParseError(
                        f"operation {operation.name} has node count 2, but site " f"has {len(site.node_ids)} node_ids"
                    )

                operation_edge = get_edge(device, site.node_ids[0], site.node_ids[1])
                edge_id = make_edge_id(site.node_ids[0], site.node_ids[1])
                if operation_edge is None:
                    raise QCSISAParseError(
                        f"operation {operation.name} has site {site.node_ids}, but edge {edge_id} "
                        f"not declared in architecture"
                    )

                if operation.name in edge_operations_seen[edge_id]:
                    continue
                edge_operations_seen[edge_id].add(operation.name)

                operation_edge.gates.extend(_transform_edge_operation_to_gates(operation.name, site.characteristics))

            else:
                raise QCSISAParseError(f"unexpected operation node count: {operation.node_count}")
    for qubit in device.qubits.values():
        if len(qubit.gates) == 0:
            qubit.dead = True

    for edge in device.edges.values():
        if len(edge.gates) == 0:
            edge.dead = True

    return device


PERFECT_FIDELITY = 1e0
PERFECT_DURATION = 1 / 100

_operation_names_to_compiler_fidelity_default = {
    Supported2QGate.CZ: 0.89,
    Supported2QGate.ISWAP: 0.90,
    Supported2QGate.CPHASE: 0.85,
    Supported2QGate.XY: 0.86,
    Supported1QGate.RX: 0.95,
    Supported1QGate.MEASURE: 0.90,
}

_operation_names_to_compiler_duration_default = {
    Supported2QGate.CZ: 200,
    Supported2QGate.ISWAP: 200,
    Supported2QGate.CPHASE: 200,
    Supported2QGate.XY: 200,
    Supported1QGate.RX: 50,
    Supported1QGate.MEASURE: 2000,
}


def _make_measure_gates(node_id: int, characteristics: Sequence[Characteristic]) -> list[MeasureInfo]:
    duration = _operation_names_to_compiler_duration_default[Supported1QGate.MEASURE]
    fidelity = _operation_names_to_compiler_fidelity_default[Supported1QGate.MEASURE]
    for characteristic in characteristics:
        if characteristic.name == "fRO":
            fidelity = characteristic.value
            break

    return [
        MeasureInfo(
            operator=Supported1QGate.MEASURE,
            qubit=str(node_id),
            target="_",
            fidelity=fidelity,
            duration=duration,
        ),
        MeasureInfo(
            operator=Supported1QGate.MEASURE,
            qubit=str(node_id),
            target=None,
            fidelity=fidelity,
            duration=duration,
        ),
    ]


def _make_rx_gates(node_id: int, benchmarks: Sequence[Operation]) -> list[GateInfo]:
    default_duration = _operation_names_to_compiler_duration_default[Supported1QGate.RX]
    default_fidelity = _operation_names_to_compiler_fidelity_default[Supported1QGate.RX]

    gates = [
        GateInfo(
            operator=Supported1QGate.RX,
            parameters=[0.0],
            arguments=[node_id],
            fidelity=PERFECT_FIDELITY,
            duration=default_duration,
        )
    ]

    fidelity = _get_frb_sim_1q(node_id, benchmarks)
    if fidelity is None:
        fidelity = default_fidelity
    for param in [np.pi, -np.pi, np.pi / 2, -np.pi / 2]:
        gates.append(
            GateInfo(
                operator=Supported1QGate.RX,
                parameters=[param],
                arguments=[node_id],
                fidelity=fidelity,
                duration=default_duration,
            )
        )
    return gates


def _make_rz_gates(node_id: int) -> list[GateInfo]:
    return [
        GateInfo(
            operator=Supported1QGate.RZ,
            parameters=["_"],
            arguments=[node_id],
            fidelity=PERFECT_FIDELITY,
            duration=PERFECT_DURATION,
        )
    ]


def _get_frb_sim_1q(node_id: int, benchmarks: Sequence[Operation]) -> Optional[float]:
    frb_sim_1q = next(
        (benchmark for benchmark in benchmarks if benchmark.name == "randomized_benchmark_simultaneous_1q"), None
    )
    if frb_sim_1q is None:
        return None

    site = next(
        (
            characteristic
            for characteristic in frb_sim_1q.sites[0].characteristics
            if isinstance(characteristic.node_ids, list)
            and len(characteristic.node_ids) == 1
            and characteristic.node_ids[0] == node_id
        ),
        None,
    )
    if site is None:
        return None

    return site.value


def _make_wildcard_1q_gates(node_id: int) -> list[GateInfo]:
    return [
        GateInfo(
            operator="_",
            parameters=["_"],
            arguments=[node_id],
            fidelity=PERFECT_FIDELITY,
            duration=PERFECT_DURATION,
        )
    ]


def _transform_qubit_operation_to_gates(
    operation_name: str,
    node_id: int,
    characteristics: Sequence[Characteristic],
    benchmarks: Sequence[Operation],
) -> list[Union[GateInfo, MeasureInfo]]:
    if operation_name == Supported1QGate.RX:
        return cast(list[Union[GateInfo, MeasureInfo]], _make_rx_gates(node_id, benchmarks))
    elif operation_name == Supported1QGate.RZ:
        return cast(list[Union[GateInfo, MeasureInfo]], _make_rz_gates(node_id))
    elif operation_name == Supported1QGate.MEASURE:
        return cast(list[Union[GateInfo, MeasureInfo]], _make_measure_gates(node_id, characteristics))
    elif operation_name == Supported1QGate.WILDCARD:
        return cast(list[Union[GateInfo, MeasureInfo]], _make_wildcard_1q_gates(node_id))
    elif operation_name in {"I", "RESET"}:
        return []
    else:
        raise QCSISAParseError(f"Unsupported qubit operation: {operation_name}")


def _make_cz_gates(characteristics: Sequence[Characteristic]) -> list[GateInfo]:
    default_duration = _operation_names_to_compiler_duration_default[Supported2QGate.CZ]
    default_fidelity = _operation_names_to_compiler_fidelity_default[Supported2QGate.CZ]

    fidelity = default_fidelity
    for characteristic in characteristics:
        if characteristic.name == "fCZ":
            fidelity = characteristic.value
            break

    return [
        GateInfo(
            operator=Supported2QGate.CZ,
            parameters=[],
            arguments=["_", "_"],
            fidelity=fidelity,
            duration=default_duration,
        )
    ]


def _make_iswap_gates(characteristics: Sequence[Characteristic]) -> list[GateInfo]:
    default_duration = _operation_names_to_compiler_duration_default[Supported2QGate.ISWAP]
    default_fidelity = _operation_names_to_compiler_fidelity_default[Supported2QGate.ISWAP]

    fidelity = default_fidelity
    for characteristic in characteristics:
        if characteristic.name == "fISWAP":
            fidelity = characteristic.value
            break

    return [
        GateInfo(
            operator=Supported2QGate.ISWAP,
            parameters=[],
            arguments=["_", "_"],
            fidelity=fidelity,
            duration=default_duration,
        )
    ]


def _make_cphase_gates(characteristics: Sequence[Characteristic]) -> list[GateInfo]:
    default_duration = _operation_names_to_compiler_duration_default[Supported2QGate.CPHASE]
    default_fidelity = _operation_names_to_compiler_fidelity_default[Supported2QGate.CPHASE]

    fidelity = default_fidelity
    for characteristic in characteristics:
        if characteristic.name == "fCPHASE":
            fidelity = characteristic.value
            break

    return [
        GateInfo(
            operator=Supported2QGate.CPHASE,
            parameters=["theta"],
            arguments=["_", "_"],
            fidelity=fidelity,
            duration=default_duration,
        )
    ]


def _make_xy_gates(characteristics: Sequence[Characteristic]) -> list[GateInfo]:
    default_duration = _operation_names_to_compiler_duration_default[Supported2QGate.XY]
    default_fidelity = _operation_names_to_compiler_fidelity_default[Supported2QGate.XY]

    fidelity = default_fidelity
    for characteristic in characteristics:
        if characteristic.name == "fXY":
            fidelity = characteristic.value
            break

    return [
        GateInfo(
            operator=Supported2QGate.XY,
            parameters=["theta"],
            arguments=["_", "_"],
            fidelity=fidelity,
            duration=default_duration,
        )
    ]


def _make_wildcard_2q_gates() -> list[GateInfo]:
    return [
        GateInfo(
            operator="_",
            parameters=["_"],
            arguments=["_", "_"],
            fidelity=PERFECT_FIDELITY,
            duration=PERFECT_DURATION,
        )
    ]


def _transform_edge_operation_to_gates(
    operation_name: str,
    characteristics: Sequence[Characteristic],
) -> list[GateInfo]:
    if operation_name == Supported2QGate.CZ:
        return _make_cz_gates(characteristics)
    elif operation_name == Supported2QGate.ISWAP:
        return _make_iswap_gates(characteristics)
    elif operation_name == Supported2QGate.CPHASE:
        return _make_cphase_gates(characteristics)
    elif operation_name == Supported2QGate.XY:
        return _make_xy_gates(characteristics)
    elif operation_name == Supported2QGate.WILDCARD:
        return _make_wildcard_2q_gates()
    else:
        raise QCSISAParseError(f"Unsupported edge operation: {operation_name}")
