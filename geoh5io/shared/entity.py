import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoh5io import shared


class Entity(ABC):
    def __init__(self, name: str, uid: uuid.UUID = None):
        self._uid = uid if uid is not None else uuid.uuid4()
        self._name = self.fix_up_name(name)
        self._visible = True
        self._allow_delete = True
        self._allow_rename = True
        self._is_public = True

    @property
    def uid(self) -> uuid.UUID:
        return self._uid

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = self.fix_up_name(new_name)

    @property
    def visible(self) -> bool:
        return self._visible

    @property
    def allow_delete(self) -> bool:
        return self._allow_delete

    @property
    def allow_rename(self) -> bool:
        return self._allow_rename

    @property
    def is_public(self) -> bool:
        return self._is_public

    @classmethod
    def fix_up_name(cls, name: str) -> str:
        """ If the given  name is not a valid one, transforms it to make it valid
        :return: a valid name built from the given name. It simply returns the given name
        if it was already valid.
        """
        # TODO: implement an actual fixup
        #  (possibly it has to be abstract with different implementations per Entity type)
        return name

    @abstractmethod
    def get_type(self) -> "shared.EntityType":
        ...
