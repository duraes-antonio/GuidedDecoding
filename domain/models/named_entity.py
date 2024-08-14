from typing import TypeVar

from typing_extensions import TypedDict

T = TypeVar('T')


class Entity(TypedDict):
    id: str


class NamedEntity(Entity):
    name: str
