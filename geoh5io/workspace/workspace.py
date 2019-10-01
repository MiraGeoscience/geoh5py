from __future__ import annotations

import inspect
import uuid
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Type, cast
from weakref import ReferenceType

from geoh5io import objects
from geoh5io.io import H5Reader
from geoh5io.objects import ObjectBase
from geoh5io.shared import weakref_utils

from .root_group import RootGroup

if TYPE_CHECKING:
    from geoh5io.groups import group
    from geoh5io.objects import object_base
    from geoh5io.data import data
    from geoh5io.shared import entity_type


@dataclass
class WorkspaceAttributes:
    contributors = None
    distance_unit = None
    ga_version = None
    version = None


class Workspace:

    _active_ref: ClassVar[ReferenceType[Workspace]] = type(None)  # type: ignore

    def __init__(self, h5file: str = None, root: RootGroup = None):
        self._workspace_attributes = None
        self._base = "GEOSCIENCE"
        self._h5file = h5file
        self._tree: Dict = {}
        self._types: Dict[uuid.UUID, ReferenceType[entity_type.EntityType]] = {}
        self._groups: Dict[uuid.UUID, ReferenceType[group.Group]] = {}
        self._objects: Dict[uuid.UUID, ReferenceType[object_base.ObjectBase]] = {}
        self._data: Dict[uuid.UUID, ReferenceType[data.Data]] = {}

        self._root = root if root is not None else RootGroup(self)

    @property
    def version(self):
        if getattr(self, "_workspace_attributes", None) is None:
            self.get_workspace_attributes()

        return (
            self._workspace_attributes.version,
            self._workspace_attributes.ga_version,
        )

    @property
    def tree(self):
        if not getattr(self, "_tree"):
            self._tree = H5Reader.get_project_tree(self.h5file, self._base)

        return self._tree

    @property
    def list_objects(self):
        """
        :return: List of object names
        """
        return [elem["name"] for elem in list(self.tree["objects"].values())]

    def get_object(self, name: str) -> List[Optional[ObjectBase]]:
        """Retrieve an object from its name

        :param name: List of object identifiers of type 'str' | 'uuid'
        :return: object_base.ObjectBase
        """

        # Extract all objects uuid with matching name
        object_uuids = [
            key
            for key in list(self.tree["objects"].keys())
            if self.tree["objects"][key]["name"] == name
        ]

        object_list: List[Optional[ObjectBase]] = []
        for uid in object_uuids:

            # Check if an object already exists in the workspace
            if self.find_object(uuid.UUID(uid)) is not None:
                object_list += [self.find_object(uuid.UUID(uid))]
                continue

            # If not, check the type
            obj_type = uuid.UUID(self.tree["objects"][uid]["type"])

            created_object: Optional[ObjectBase] = None
            for _, member in inspect.getmembers(objects):

                if (
                    inspect.isclass(member)
                    and issubclass(member, ObjectBase)
                    and member is not ObjectBase
                    and member.default_type_uid() == obj_type
                ):
                    known_type = member.find_or_create_type(self)
                    created_object = member(
                        known_type, self.tree["objects"][uid]["name"], uuid.UUID(uid)
                    )

            # Object of unknown type
            if created_object is None:
                assert RuntimeError("Only objects of known type have been implemented")
            #             unknown_type =

            object_list += [created_object]

        return object_list

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
        # print(entity_type.uid, entity_type)
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

    @property
    def h5file(self) -> str:
        assert self._h5file is not None, "The 'h5file' property name must be set"
        return self._h5file

    @h5file.setter
    def h5file(self, h5file):
        self._h5file = h5file

    def get_workspace_attributes(self):
        """ Fetch the workspace attributes
        """

        if getattr(self, "_project_attributes", None) is None:

            self._workspace_attributes = WorkspaceAttributes()

            attributes = H5Reader.get_project_attributes(self.h5file, self._base)

            for (attr, value) in zip(attributes.keys(), attributes.values()):
                setattr(self._workspace_attributes, attr, value)

        return self._workspace_attributes

    def load_geoh5_workspace(self):
        """ Load the groups, objects, data and types from H5file
        """

        tree = H5Reader.get_project_tree(self.h5file, self._base)
        # if getattr(self, "_project_attributes", None) is None:
        # for (attr, value) in zip(attributes.keys(), attributes.values()):
        #     setattr(self._workspace_attributes, attr, value)

        return tree


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
