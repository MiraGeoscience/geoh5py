from __future__ import annotations

import inspect
import uuid
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Type, cast
from weakref import ReferenceType

from geoh5io import data, groups, objects
from geoh5io.data import Data
from geoh5io.groups import Group
from geoh5io.io import H5Reader
from geoh5io.objects import ObjectBase
from geoh5io.shared import Coord3D, weakref_utils
from geoh5io.shared.entity import Entity

from .root_group import RootGroup

if TYPE_CHECKING:
    from geoh5io.groups import group
    from geoh5io.objects import object_base
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
    def list_groups(self):
        """
        :return: List of names of groups
        """
        return [
            value["name"]
            for value in self.tree.values()
            if value["entity_type"] == "group"
        ]

    @property
    def list_objects(self):
        """
        :return: List of names of objects
        """
        return [
            value["name"]
            for value in self.tree.values()
            if value["entity_type"] == "object"
        ]

    @property
    def list_data(self):
        """
        :return: List of names of data
        """
        return [
            value["name"]
            for value in self.tree.values()
            if value["entity_type"] == "data"
        ]

    # @property
    # def get_group_types_list(self):
    #     """
    #     :return: List of names of group types
    #     """
    #     return [value for value in self.tree["types"]["group"].values()]
    #
    # @property
    # def get_object_types_list(self):
    #     """
    #     :return: List of names of object types
    #     """
    #     return [value for value in self.tree["types"]["object"].values()]
    #
    # @property
    # def get_data_types_list(self):
    #     """
    #     :return: List of names of data types
    #     """
    #     return [value for value in self.tree["types"]["data"].values()]

    def get_entity(self, name: str) -> List[Optional[Entity]]:
        """Retrieve an entity from its name

        :param name: List of object identifiers of type 'str' | 'uuid'
        :return: object_base.ObjectBase
        """

        base_classes = {"group": Group, "object": ObjectBase, "data": Data}

        if isinstance(name, uuid.UUID):
            list_entity_uid = [name]

        else:  # Extract all objects uuid with matching name
            list_entity_uid = [
                key
                for key in self.tree
                if (self.tree[key]["name"] == name)
                and ("type" not in self.tree[key]["entity_type"])
            ]

        entity_list: List[Optional[Entity]] = []
        for uid in list_entity_uid:
            entity_type = self.tree[uid]["entity_type"]

            # Check if an object already exists in the workspace
            finder = getattr(self, f"find_{entity_type}")
            if finder(uid) is not None:
                entity_list += [finder(uid)]
                continue

            created_object = self.create_entity(
                base_classes[entity_type],
                self.tree[uid]["type"],
                self.tree[uid]["name"],
                uid,
                attributes=self.tree[uid],
                type_attributes=self.tree[self.tree[uid]["type"]],
            )

            # Object of unknown type
            if created_object is None:
                assert RuntimeError("Only objects of known type have been implemented")
            #             unknown_type =

            entity_list += [created_object]

        return entity_list

    def create_entity(
        self,
        entity_class,
        entity_type_uid: uuid.UUID,
        name: str,
        uid: uuid.UUID,
        attributes=None,
        type_attributes=None,
    ) -> Optional[Entity]:

        created_entity: Optional[Entity] = None

        if entity_class is not Data:
            for _, member in inspect.getmembers(groups) + inspect.getmembers(objects):

                if (
                    inspect.isclass(member)
                    and issubclass(member, entity_class)
                    and member is not entity_class
                    and hasattr(member, "default_type_uid")
                    and member.default_type_uid() is not None
                    and member.default_type_uid() == entity_type_uid
                ):

                    known_type = member.find_or_create_type(self)
                    created_entity = member(known_type, name, uid)

            # Special case for CustomGroup without uuid
            if (created_entity is None) and entity_class == Group:
                custom = groups.custom_group.CustomGroup
                unknown_type = custom.find_or_create_type(self)
                created_entity = custom(unknown_type, name, uid)

        else:

            data_type = data.data_type.DataType(
                self,
                entity_type_uid,
                getattr(
                    data.primitive_type_enum.PrimitiveTypeEnum,
                    type_attributes["primitive_type"].upper(),
                ),
            )

            for _, member in inspect.getmembers(data):

                if (
                    inspect.isclass(member)
                    and issubclass(member, entity_class)
                    and member is not entity_class
                    and hasattr(member, "primitive_type")
                    and inspect.ismethod(member.primitive_type)
                    and data_type.primitive_type is member.primitive_type()
                ):

                    created_entity = member(
                        data_type,
                        getattr(
                            data.data_association_enum.DataAssociationEnum,
                            attributes["association"].upper(),
                        ),
                        name,
                        uid,
                    )

        return created_entity

    def get_children_list(self, uid: uuid.UUID, children_type: str):
        """

        :param uid: UUID of object
        :param children_type:
        :return:
        """

        name_list: List[str] = []

        for key in self.tree[uid]["children"]:

            if self.tree[key]["entity_type"] == children_type:
                name_list += [self.tree[key]["name"]]

        return name_list

    def get_children(self, uid: uuid.UUID, name: str) -> List[Optional[Entity]]:
        """
        Return a data object

        :param: uid: UUID of object
        :param name: Name of children
        :return: List[entity]: A list of registered objects
        """

        if isinstance(name, uuid.UUID):
            return self.get_entity(name)

        for key in self.tree[uid]["children"]:

            if self.tree[key]["name"] == name:

                return self.get_entity(key)

        return []

    def get_parent(self, uid: uuid.UUID) -> List[Optional[Entity]]:
        """
        Return the parent object

        :param: uid: UUID of object
        :return: entity: The registered parent entity
        """

        if self.tree[uid]["parent"]:
            return self.get_entity(self.tree[uid]["parent"])

        return []

    def get_data_value(self, uid: uuid.UUID) -> Optional[float]:
        """
        Get the data values from the source h5 file

        :param uid: UUID of target data object
        :return: value: Data value
        """

        return H5Reader.get_value(self._h5file, self._base, uid)

    def get_vertices(self, uid: uuid.UUID) -> Coord3D:
        """
        Get the vertices of an object from the source h5 file

        :param uid: UUID of target entity
        :return: value: Data value
        """

        return H5Reader.get_vertices(self._h5file, self._base, uid)

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

    def _register_data(self, data_obj: "data.Data"):
        weakref_utils.insert_once(self._data, data_obj.uid, data_obj)

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
