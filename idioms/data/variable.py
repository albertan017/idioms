"""Information about variables in a function

Taken from the replication package for DIRTY: Chen, Qibin, et al. "Augmenting decompiler output with learned variable names and types." 31st USENIX Security Symposium (USENIX Security 22). 2022.
"""

from json import dumps
from typing import Any

# Huge hack to get importing to work with the decompiler
try:
    from idioms.data.types import TypeLibCodec, TypeInfo, PlaceholderType
except ImportError:
    from .types import TypeLibCodec, TypeInfo, PlaceholderType

class Location:
    """A variable location"""
    def json_key(self) -> str:
        """Returns a string suitable as a key in a JSON dict"""
        raise NotImplementedError()


class Register(Location):
    """A register

    name: the name of the register
    """

    def __init__(self, name: int):
        self.name = name

    def json_key(self) -> str:
        return f"r{self.name}"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Register) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"Reg {self.name}"


class Stack(Location):
    """A location on the stack

    offset: the offset from the base pointer
    """

    def __init__(self, offset: int):
        self.offset = offset

    def json_key(self) -> str:
        return f"s{self.offset}"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Stack) and self.offset == other.offset

    def __hash__(self) -> int:
        return hash(self.offset)

    def __repr__(self) -> str:
        return f"Stk 0x{self.offset:x}"


def location_from_json_key(key: str) -> "Location":
    """Hacky way to return a location from a JSON key"""
    if key.startswith("s"):
        return Stack(int(key[1:]))
    else:
        return Register(key[1:])

class Variable:
    """A variable

    typ: the type of the variable
    name: an optional user-defined name for the variable
    user: true if the name is user-defined
    """

    def __init__(self, typ: TypeInfo, name: str, user: bool):
        self.typ = typ
        self.name = name
        self.user = user

    def to_json(self):
        return {
            "t": self.typ._to_json(),
            "n": self.name,
            "u": self.user,
        }

    @classmethod
    def from_json(cls, d):
        try:
            typ = TypeLibCodec.decode(dumps(d["t"]))
        except KeyError:
            typ = PlaceholderType()
        return cls(typ=typ, name=d["n"], user=d["u"])

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Variable)
            and self.name == other.name
            and self.typ == other.typ
        )

    def __hash__(self):
        return hash((self.name, self.typ))

    def __repr__(self) -> str:
        name_source = "U" if self.user else "A"
        return f"{str(self.typ)} {self.name} ({name_source})"
