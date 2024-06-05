"""Models and functions that support the use of the `quilc` rpcq server."""

import json
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from deprecated.sphinx import deprecated
from rpcq.messages import TargetDevice as TargetQuantumProcessor

JsonValue = Union[type(None), bool, int, float, str, list["JsonValue"], dict[str, "JsonValue"]]


@dataclass
class Operator:
    """Operator class for representing a quantum gate or measurement."""

    operator: Optional[str] = None
    duration: Optional[float] = None
    fidelity: Optional[float] = None

    def __post_init__(self):
        self.duration = float(self.duration) if self.duration is not None else None
        self.fidelity = float(self.fidelity) if self.fidelity is not None else None

    def _dict(self) -> dict[str, JsonValue]:
        if type(self) is Operator:
            raise ValueError("Should be a subclass")
        return dict(
            operator=self.operator,
            duration=self.duration,
            fidelity=self.fidelity,
        )

    @deprecated(
        version="4.6.2",
        reason="No longer requires serialization of RPCQ objects and is dropping Pydantic as a dependency.",  # noqa: E501
    )
    def dict(self):
        """Return a dictionary representation of the Operator."""
        return self._dict()

    @classmethod
    def _parse_obj(cls, dictionary: dict):
        return Operator(**dictionary)

    @classmethod
    @deprecated(
        version="4.6.2",
        reason="No longer requires serialization of RPCQ objects and is dropping Pydantic as a dependency.",  # noqa: E501
    )
    def parse_obj(cls, dictionary: dict):
        """Parse a dictionary into an Operator object."""
        return cls._parse_obj(dictionary)


@dataclass
class MeasureInfo(Operator):
    """MeasureInfo class for representing a measurement operation."""

    qubit: Optional[Union[int, str]] = None
    target: Optional[Union[int, str]] = None
    operator_type: Literal["measure"] = "measure"

    def __post_init__(self):
        """Post initialization method for MeasureInfo."""
        self.qubit = str(self.qubit)

    def _dict(self) -> dict[str, JsonValue]:
        return dict(
            operator_type=self.operator_type,
            operator=self.operator,
            duration=self.duration,
            fidelity=self.fidelity,
            qubit=self.qubit,
            target=self.target,
        )

    @deprecated(
        version="4.7",
        reason="No longer requires serialization of RPCQ objects and is dropping Pydantic as a dependency.",  # noqa: E501
    )
    def dict(self):
        """Return a dictionary representation of the MeasureInfo."""
        return self._dict()

    @classmethod
    def _parse_obj(cls, dictionary: dict) -> "MeasureInfo":
        return MeasureInfo(
            operator=dictionary.get("operator"),
            duration=dictionary.get("duration"),
            fidelity=dictionary.get("fidelity"),
            qubit=dictionary.get("qubit"),
            target=dictionary.get("target"),
        )

    @classmethod
    @deprecated(
        version="4.6.2",
        reason="No longer requires serialization of RPCQ objects and is dropping Pydantic as a dependency.",  # noqa: E501
    )
    def parse_obj(cls, dictionary: dict):
        """Parse a dictionary into a MeasureInfo object."""
        return cls._parse_obj(dictionary)


@dataclass
class GateInfo(Operator):
    """GateInfo class for representing a quantum gate operation."""

    parameters: list[Union[float, str]] = field(default_factory=list)
    arguments: list[Union[int, str]] = field(default_factory=list)
    operator_type: Literal["gate"] = "gate"

    def _dict(self) -> dict[str, JsonValue]:
        return dict(
            operator_type=self.operator_type,
            operator=self.operator,
            duration=self.duration,
            fidelity=self.fidelity,
            parameters=self.parameters,
            arguments=self.arguments,
        )

    @deprecated(
        version="4.6.2",
        reason="No longer requires serialization of RPCQ objects and is dropping Pydantic as a dependency.",  # noqa: E501
    )
    def dict(self):
        """Return a dictionary representation of the GateInfo."""
        return self._dict()

    @classmethod
    def _parse_obj(cls, dictionary: dict) -> "GateInfo":
        return GateInfo(
            operator=dictionary.get("operator"),
            duration=dictionary.get("duration"),
            fidelity=dictionary.get("fidelity"),
            parameters=dictionary["parameters"],
            arguments=dictionary["arguments"],
        )

    @classmethod
    @deprecated(
        version="4.6.2",
        reason="No longer requires serialization of RPCQ objects and is dropping Pydantic as a dependency.",  # noqa: E501
    )
    def parse_obj(cls, dictionary: dict):
        """Parse a dictionary into a GateInfo object."""
        return cls._parse_obj(dictionary)


def _parse_operator(dictionary: dict) -> Union[GateInfo, MeasureInfo]:
    operator_type = dictionary["operator_type"]
    if operator_type == "measure":
        return MeasureInfo._parse_obj(dictionary)
    if operator_type == "gate":
        return GateInfo._parse_obj(dictionary)
    raise ValueError("Should be a subclass of Operator")


@dataclass
class Qubit:
    """Qubit class for representing a qubit in a quantum processor."""

    id: int
    dead: Optional[bool] = False
    gates: list[Union[GateInfo, MeasureInfo]] = field(default_factory=list)

    def _dict(self) -> dict[str, JsonValue]:
        encoding = dict(id=self.id, gates=[g._dict() for g in self.gates])
        if self.dead:
            encoding["dead"] = self.dead
        return encoding

    @deprecated(
        version="4.6.2",
        reason="No longer requires serialization of RPCQ objects and is dropping Pydantic as a dependency.",  # noqa: E501
    )
    def dict(self):
        """Return a dictionary representation of the Qubit."""
        return self._dict()

    @classmethod
    def _parse_obj(cls, dictionary: dict) -> "Qubit":
        return Qubit(
            id=dictionary["id"],
            dead=bool(dictionary.get("dead")),
            gates=[_parse_operator(v) for v in dictionary.get("gates", [])],
        )

    @classmethod
    @deprecated(
        version="4.6.2",
        reason="No longer requires serialization of RPCQ objects and is dropping Pydantic as a dependency.",  # noqa: E501
    )
    def parse_obj(cls, dictionary: dict):
        """Parse a dictionary into a Qubit object."""
        return cls._parse_obj(dictionary)


@dataclass
class Edge:
    """Edge class for representing a connection between two qubits."""

    ids: list[int]
    dead: Optional[bool] = False
    gates: list[GateInfo] = field(default_factory=list)

    def _dict(self) -> dict[str, JsonValue]:
        encoding = dict(ids=self.ids, gates=[g._dict() for g in self.gates])
        if self.dead:
            encoding["dead"] = self.dead
        return encoding

    @deprecated(
        version="4.6.2",
        reason="No longer requires serialization of RPCQ objects and is dropping Pydantic as a dependency.",  # noqa: E501
    )
    def dict(self):
        """Return a dictionary representation of the Edge."""
        return self._dict()

    @classmethod
    def _parse_obj(cls, dictionary: dict) -> "Edge":
        return Edge(
            ids=dictionary["ids"],
            dead=bool(dictionary.get("dead")),
            gates=[GateInfo._parse_obj(g) for g in dictionary.get("gates", [])],
        )

    @classmethod
    @deprecated(
        version="4.6.2",
        reason="No longer requires serialization of RPCQ objects and is dropping Pydantic as a dependency.",  # noqa: E501
    )
    def parse_obj(cls, dictionary: dict):
        """Parse a dictionary into an Edge object."""
        return cls._parse_obj(dictionary)


@dataclass
class CompilerISA:
    """CompilerISA class for representing the instruction set architecture of a quantum processor."""

    qubits: dict[str, Qubit] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)

    def _dict(self, by_alias=False) -> dict[str, JsonValue]:
        return {
            "1Q" if by_alias else "qubits": {k: q._dict() for k, q in self.qubits.items()},
            "2Q" if by_alias else "edges": {k: e._dict() for k, e in self.edges.items()},
        }

    @deprecated(
        version="4.6.2",
        reason="No longer requires serialization of RPCQ objects and is dropping Pydantic as a dependency.",  # noqa: E501
    )
    def dict(self, by_alias=False):
        """Return a dictionary representation of the CompilerISA."""
        return self._dict(by_alias=by_alias)

    @classmethod
    def _parse_obj(cls, dictionary: dict):
        qubit_dict = dictionary.get("1Q", {})
        edge_dict = dictionary.get("2Q", {})
        return CompilerISA(
            qubits={k: Qubit._parse_obj(v) for k, v in qubit_dict.items()},
            edges={k: Edge._parse_obj(v) for k, v in edge_dict.items()},
        )

    @classmethod
    @deprecated(
        version="4.6.2",
        reason="No longer requires serialization of RPCQ objects and is dropping Pydantic as a dependency.",  # noqa: E501
    )
    def parse_obj(cls, dictionary: dict):
        """Parse a dictionary into a CompilerISA object."""
        return cls._parse_obj(dictionary)

    @classmethod
    @deprecated(
        version="4.6.2",
        reason="No longer requires serialization of RPCQ objects and is dropping Pydantic as a dependency.",  # noqa: E501
    )
    def parse_file(cls, filename: str):
        """Parse a JSON file into a CompilerISA object."""
        with open(filename) as file:
            json_dict = json.load(file)
            return cls._parse_obj(json_dict)


def add_qubit(quantum_processor: CompilerISA, node_id: int) -> Qubit:
    """Add a qubit to the quantum processor ISA."""
    if node_id not in quantum_processor.qubits:
        quantum_processor.qubits[str(node_id)] = Qubit(id=node_id)
    return quantum_processor.qubits[str(node_id)]


def get_qubit(quantum_processor: CompilerISA, node_id: int) -> Optional[Qubit]:
    """Get a qubit from the quantum processor ISA."""
    return quantum_processor.qubits.get(str(node_id))


def make_edge_id(qubit1: int, qubit2: int) -> str:
    """Make an edge ID from two qubit IDs."""
    return "-".join([str(qubit) for qubit in sorted([qubit1, qubit2])])


def add_edge(quantum_processor: CompilerISA, qubit1: int, qubit2: int) -> Edge:
    """Add an Edge between two qubit IDs."""
    edge_id = make_edge_id(qubit1, qubit2)
    if edge_id not in quantum_processor.edges:
        quantum_processor.edges[edge_id] = Edge(ids=sorted([qubit1, qubit2]))
    return quantum_processor.edges[edge_id]


def get_edge(quantum_processor: CompilerISA, qubit1: int, qubit2: int) -> Optional[Edge]:
    """Get an Edge between two qubit IDs."""
    edge_id = make_edge_id(qubit1, qubit2)
    return quantum_processor.edges.get(edge_id)


def compiler_isa_to_target_quantum_processor(compiler_isa: CompilerISA) -> TargetQuantumProcessor:
    """Convert a CompilerISA object to a TargetQuantumProcessor object."""
    return TargetQuantumProcessor(isa=compiler_isa.dict(by_alias=True), specs={})


class Supported1QGate:
    """Supported 1Q gates."""

    I = "I"  # noqa: E741 - I is not an ambiguous gate name.
    RX = "RX"
    RZ = "RZ"
    MEASURE = "MEASURE"
    WILDCARD = "WILDCARD"


class Supported2QGate:
    """Supported 2Q gates."""

    WILDCARD = "WILDCARD"
    CZ = "CZ"
    ISWAP = "ISWAP"
    CPHASE = "CPHASE"
    XY = "XY"
