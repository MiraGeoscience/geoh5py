from __future__ import annotations

import uuid
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Type, cast
from weakref import ReferenceType

from geoh5io.shared import weakref_utils

from .root_group import RootGroup

if TYPE_CHECKING:
    from geoh5io.groups import group
    from geoh5io.objects import object_base
    from geoh5io.data import data
    from geoh5io.shared import entity_type


class Workspace:

    _active_ref: ClassVar[ReferenceType[Workspace]] = type(None)  # type: ignore

    def __init__(self, root: RootGroup = None):
        self.version = None
        self._distance_unit = None
        self._contributors: List[str] = []

        self._types: Dict[uuid.UUID, ReferenceType[entity_type.EntityType]] = {}
        self._groups: Dict[uuid.UUID, ReferenceType[group.Group]] = {}
        self._objects: Dict[uuid.UUID, ReferenceType[object_base.ObjectBase]] = {}
        self._data: Dict[uuid.UUID, ReferenceType[data.Data]] = {}

        self._root = root if root is not None else RootGroup(self)

    @property
    def root(self) -> "group.Group":
        return self._root

    def activate(self):
        """ Makes this workspace the active one.

            In case the workspace gets deleted, Workspace.active() safely returns None.
        """
        if Workspace._active_ref() is not self:
            Workspace._active_ref = weakref.ref(self)

    def deactivate(self):
        """ Deactivate this workspace if it was the active one, else does nothing.
        """
        if Workspace._active_ref() is self:
            Workspace._active_ref = type(None)

    @staticmethod
    def active() -> Workspace:
        """ Get the active workspace. """
        active_one = Workspace._active_ref()
        if active_one is None:
            raise RuntimeError("No active workspace.")

        # so that type check does not complain of possible returned None
        return cast(Workspace, active_one)

    def _register_type(self, entity_type: "entity_type.EntityType"):
        weakref_utils.insert_once(self._types, entity_type.uid, entity_type)

    def _register_group(self, group: "group.Group"):
        weakref_utils.insert_once(self._groups, group.uid, group)

    def _register_data(self, data: "data.Data"):
        weakref_utils.insert_once(self._data, data.uid, data)

    def _register_object(self, obj: "object_base.ObjectBase"):
        weakref_utils.insert_once(self._objects, obj.uid, obj)

    def find_type(
        self, type_uid: uuid.UUID, type_class: Type["entity_type.EntityType"]
    ) -> Optional["entity_type.EntityType"]:
        found_type = weakref_utils.get_clean_ref(self._types, type_uid)
        return found_type if isinstance(found_type, type_class) else None

    def all_groups(self) -> List["group.Group"]:
        weakref_utils.remove_none_referents(self._groups)
        return [cast("group.Group", v()) for v in self._groups.values()]

    def find_group(self, group_uid: uuid.UUID) -> Optional["group.Group"]:
        return weakref_utils.get_clean_ref(self._groups, group_uid)

    def all_objects(self) -> List["object_base.ObjectBase"]:
        weakref_utils.remove_none_referents(self._objects)
        return [cast("object_base.ObjectBase", v()) for v in self._objects.values()]

    def find_object(self, object_uid: uuid.UUID) -> Optional["object_base.ObjectBase"]:
        return weakref_utils.get_clean_ref(self._objects, object_uid)

    def all_data(self) -> List["data.Data"]:
        weakref_utils.remove_none_referents(self._data)
        return [cast("data.Data", v()) for v in self._data.values()]

    def find_data(self, data_uid: uuid.UUID) -> Optional["data.Data"]:
        return weakref_utils.get_clean_ref(self._data, data_uid)


@contextmanager
def active_workspace(workspace: Workspace):
    previous_active_ref = Workspace._active_ref  # pylint: disable=protected-access
    workspace.activate()
    yield workspace

    workspace.deactivate()
    # restore previous active workspace when leaving the context
    previous_active = previous_active_ref()
    if previous_active is not None:
        previous_active.activate()  # pylint: disable=no-member
