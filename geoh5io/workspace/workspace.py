from __future__ import annotations

import uuid
import weakref
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    Dict,
    Optional,
    Type,
    Union,
    ValuesView,
    cast,
)

from geoh5io.io import H5Reader

from .root_group import RootGroup

if TYPE_CHECKING:
    from geoh5io.groups import group
    from geoh5io.objects import object_base
    from geoh5io.data import data
    from geoh5io.shared import entity_type


WeakRefDuckType = Union[weakref.ReferenceType, Callable[[], Optional["Workspace"]]]


class Workspace:

    _active_ref: ClassVar[WeakRefDuckType] = type(None)

    def __init__(self, root: RootGroup = None):
        self._project_attributes: Dict = {
            "version": None,
            "distance_unit": None,
            "contributors": [],
        }
        self._base = "GEOSCIENCE"
        self._h5file = None
        self._h5reader = H5Reader()

        # TODO: store values as weak references
        self._types: Dict[uuid.UUID, entity_type.EntityType] = {}
        self._groups: Dict[uuid.UUID, group.Group] = {}
        self._objects: Dict[uuid.UUID, object_base.ObjectBase] = {}
        self._data: Dict[uuid.UUID, data.Data] = {}

        self._root = root if root is not None else RootGroup(self)

    @property
    def h5reader(self) -> H5Reader:

        if getattr(self, "_h5reader", None) is None:
            self._h5reader = H5Reader(h5file=self._h5file)

        return self._h5reader

    @property
    def version(self):

        if getattr(self, "_project_attributes", None) is None:
            self._project_attributes = self._h5reader.get_project_attributes()

        return self._project_attributes["version"]

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

    # pylint: disable=redefined-outer-name
    def register_type(self, staged_type: "entity_type.EntityType"):
        # TODO: raise exception if it does already exists
        self._types[staged_type.uid] = staged_type

    def register_group(self, staged_group: "group.Group"):
        # TODO: raise exception if it does already exists
        self._groups[staged_group.uid] = staged_group

    def register_data(self, staged_data: "data.Data"):
        # TODO: raise exception if it does already exists
        self._data[staged_data.uid] = staged_data

    def register_object(self, staged_object: "object_base.ObjectBase"):
        # TODO: raise exception if it does already exists
        self._objects[staged_object.uid] = staged_object

    def find_type(
        self, type_uid: uuid.UUID, type_class: Type["entity_type.EntityType"]
    ) -> Optional["entity_type.EntityType"]:
        found_type = self._types.get(type_uid, None)
        if found_type is not None and isinstance(found_type, type_class):
            return found_type

        return None

    def all_groups(self) -> ValuesView["group.Group"]:
        return self._groups.values()

    def find_group(self, group_uid: uuid.UUID) -> Optional["group.Group"]:
        return self._groups.get(group_uid, None)

    def all_objects(self) -> ValuesView["object_base.ObjectBase"]:
        return self._objects.values()

    def find_object(self, object_uid: uuid.UUID) -> Optional["object_base.ObjectBase"]:
        return self._objects.get(object_uid, None)

    def all_data(self) -> ValuesView["data.Data"]:
        return self._data.values()

    def find_data(self, data_uid: uuid.UUID) -> Optional["data.Data"]:
        return self._data.get(data_uid, None)

    # @property
    # def project_attributes(self):
    #
    #     if getattr(self, "_project_attrs", None) is None:
    #         self._version = H5Reader.get_project_attributes(self._h5file, self._base)
    #
    #     return self._project_attrs

    @property
    def h5file(self) -> str:
        assert self._h5file is not None, "The 'h5file' property name must be set"
        return self._h5file

    # @h5file.setter
    # def h5file(self, h5file: str):
    #     self._h5file = h5file


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
