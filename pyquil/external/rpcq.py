import json
from typing import Dict, List, Union, Optional, Any
from rpcq.messages import TargetDevice as TargetQuantumProcessor
from dataclasses import dataclass, field

JsonValue = Union[type(None), bool, int, float, str, List["JsonValue"], Dict[str, "JsonValue"]]
#JsonValue = Any

@dataclass
class Operator:
    operator: Optional[str] = None
    duration: Optional[float] = None
    fidelity: Optional[float] = None

    def __post_init__(self):
        self.duration = float(self.duration) if self.duration is not None else None
        self.fidelity = float(self.fidelity) if self.fidelity is not None else None

    def to_dict(self) -> Dict[str, JsonValue]:
        if type(self) is Operator:
            raise ValueError("Should be a subclass")
        return dict(
            operator=self.operator,
            duration=self.duration,
            fidelity=self.fidelity,
        )

    @classmethod
    def from_dict(cls, dictionary):
        operator_type = dictionary["operator_type"]
        if operator_type == "measure":
            return MeasureInfo.from_dict(dictionary)
        if operator_type == "gate":
            return GateInfo.from_dict(dictionary)
        raise ValueError("Should be a subclass")


@dataclass
class MeasureInfo(Operator):
    qubit: Optional[Union[int, str]] = None
    target: Optional[Union[int, str]] = None

    def __post_init__(self):
        self.qubit = str(self.qubit)

    def to_dict(self) -> Dict[str, JsonValue]:
        encoding = super().to_dict()
        encoding["operator_type"] = "measure"
        encoding["qubit"] = self.qubit
        encoding["target"] = self.target
        return encoding

    @classmethod
    def from_dict(cls, dictionary: dict) -> "MeasureInfo":
        return MeasureInfo(
            operator=dictionary.get("operator"),
            duration=dictionary.get("duration"),
            fidelity=dictionary.get("fidelity"),
            qubit=dictionary.get("qubit"),
            target=dictionary.get("target"),
        )


@dataclass
class GateInfo(Operator):
    parameters: List[Union[float, str]] = field(default_factory=list)
    arguments: List[Union[int, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, JsonValue]:
        encoding = super().to_dict()
        encoding["operator_type"] = "gate"
        encoding["parameters"] = self.parameters
        encoding["arguments"] = self.arguments
        return encoding

    @classmethod
    def from_dict(cls, dictionary: dict) -> "GateInfo":
        return GateInfo(
            operator=dictionary.get("operator"),
            duration=dictionary.get("duration"),
            fidelity=dictionary.get("fidelity"),
            parameters=dictionary["parameters"],
            arguments=dictionary["arguments"],
        )


@dataclass
class Qubit:
    id: int
    dead: Optional[bool] = False
    gates: List[Union[GateInfo, MeasureInfo]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, JsonValue]:
        encoding = dict(
            id=self.id,
            gates=[g.to_dict() for g in self.gates]
        )
        if self.dead:
            encoding["dead"] = self.dead
        return encoding

    @classmethod
    def from_dict(cls, dictionary: dict) -> "Qubit":
        return Qubit(
            id=dictionary["id"],
            dead=bool(dictionary.get("dead")),
            gates=[Operator.from_dict(v) for v in dictionary.get("gates", [])],
        )


@dataclass
class Edge:
    ids: List[int]
    dead: Optional[bool] = False
    gates: List[GateInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, JsonValue]:
        encoding = dict(
            ids=self.ids,
            gates=[g.to_dict() for g in self.gates]
        )
        if self.dead:
            encoding["dead"] = self.dead
        return encoding

    @classmethod
    def from_dict(cls, dictionary: dict) -> "Edge":
        return Edge(
            ids=dictionary["ids"],
            dead=bool(dictionary.get("dead")),
            gates=[GateInfo.from_dict(g) for g in dictionary.get("gates", [])]
        )


@dataclass
class CompilerISA:
    qubits: Dict[str, Qubit] = field(default_factory=dict)
    edges: Dict[str, Edge] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, JsonValue]:
        return {
            "1Q": {k: q.to_dict() for k, q in self.qubits.items()},
            "2Q": {k: e.to_dict() for k, e in self.edges.items()}
        }

    @classmethod
    def from_dict(cls, dictionary: dict):
        qubit_dict = dictionary.get("1Q", {})
        edge_dict = dictionary.get("2Q", {})
        return CompilerISA(
            qubits={k: Qubit.from_dict(v) for k, v in qubit_dict.items()},
            edges={k: Edge.from_dict(v) for k, v in edge_dict.items()},
        )

    @classmethod
    def parse_file(cls, filename: str):
        with open(filename, "r") as file:
            json_dict = json.load(file)
            return cls.from_dict(json_dict)



def add_qubit(quantum_processor: CompilerISA, node_id: int) -> Qubit:
    if node_id not in quantum_processor.qubits:
        quantum_processor.qubits[str(node_id)] = Qubit(id=node_id)
    return quantum_processor.qubits[str(node_id)]


def get_qubit(quantum_processor: CompilerISA, node_id: int) -> Optional[Qubit]:
    return quantum_processor.qubits.get(str(node_id))


def make_edge_id(qubit1: int, qubit2: int) -> str:
    return "-".join([str(qubit) for qubit in sorted([qubit1, qubit2])])


def add_edge(quantum_processor: CompilerISA, qubit1: int, qubit2: int) -> Edge:
    edge_id = make_edge_id(qubit1, qubit2)
    if edge_id not in quantum_processor.edges:
        quantum_processor.edges[edge_id] = Edge(ids=sorted([qubit1, qubit2]))
    return quantum_processor.edges[edge_id]


def get_edge(quantum_processor: CompilerISA, qubit1: int, qubit2: int) -> Optional[Edge]:
    edge_id = make_edge_id(qubit1, qubit2)
    return quantum_processor.edges.get(edge_id)


def compiler_isa_to_target_quantum_processor(compiler_isa: CompilerISA) -> TargetQuantumProcessor:
    return TargetQuantumProcessor(isa=compiler_isa.to_dict(), specs={})


class Supported1QGate:
    I = "I"
    RX = "RX"
    RZ = "RZ"
    MEASURE = "MEASURE"
    WILDCARD = "WILDCARD"


class Supported2QGate:
    WILDCARD = "WILDCARD"
    CZ = "CZ"
    ISWAP = "ISWAP"
    CPHASE = "CPHASE"
    XY = "XY"
