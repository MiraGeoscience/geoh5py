from __future__ import annotations

import inspect
import os
import uuid
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Type, cast
from weakref import ReferenceType

import numpy as np

from geoh5io import data, groups, objects
from geoh5io.data import Data
from geoh5io.groups import CustomGroup, Group
from geoh5io.io import H5Reader, H5Writer
from geoh5io.objects import ObjectBase, Points
from geoh5io.shared import Coord3D, weakref_utils
from geoh5io.shared.entity import Entity

from .root_group import RootGroup

if TYPE_CHECKING:
    from geoh5io.groups import group
    from geoh5io.objects import object_base
    from geoh5io.shared import entity_type


@dataclass
class WorkspaceAttributes:
    contributors = np.asarray(["UserName"], dtype="object")
    distance_unit = "meter"
    ga_version = "1"
    version = 1.0
    name = "Workspace"


class Workspace:

    _active_ref: ClassVar[ReferenceType[Workspace]] = type(None)  # type: ignore

    def __init__(self, h5file: Optional[str] = None, root: RootGroup = None):
        self._workspace_attributes = None
        self._base = "GEOSCIENCE"
        self._tree: Dict = {}
        self._types: Dict[uuid.UUID, ReferenceType[entity_type.EntityType]] = {}
        self._groups: Dict[uuid.UUID, ReferenceType[group.Group]] = {}
        self._objects: Dict[uuid.UUID, ReferenceType[object_base.ObjectBase]] = {}
        self._data: Dict[uuid.UUID, ReferenceType[data.Data]] = {}

        self.h5file = h5file

        self._root = root if root is not None else RootGroup(self)

    @property
    def base_name(self):
        """
        @property
        base_name

        Returns
        -------
        name: str
            Name of the project ['GEOSCIENCE']
        """
        return self._base

    @property
    def version(self):
        """
        @property
        version

        Returns
        -------
        version: List(float)
            (Version, Geoscience Analyst version)
        """
        if getattr(self, "_workspace_attributes", None) is None:
            self.get_workspace_attributes()

        return (
            self._workspace_attributes["version"],
            self._workspace_attributes["ga_version"],
        )

    @property
    def tree(self):
        """
        @property
        tree

        Returns
        -------
        tree: dict
            Dictionary of entities and entity_types contained in the project.
            Used for light reference to attributes, parent and children.
            {uuid:
                {'name': value},
                {'attr1': value},
                ...
                {'parent': uuid},
                {'children': [uuid1, uuid2,....],
             ...
             }
        """
        if not getattr(self, "_tree"):
            if self.h5file is not None:
                self._tree = H5Reader.get_project_tree(self.h5file, self._base)
            else:
                self._tree = {}

        return self._tree

    @property
    def list_groups(self):
        """
        @property
        list_groups

        Returns
        -------
        groups: List[str]
            Names of all groups registered in the tree
        """
        return [
            value["name"]
            for value in self.tree.values()
            if value["entity_type"] == "group"
        ]

    @property
    def list_objects(self):
        """
        @property
        list_objects

        Returns
        -------
        objects: List[str]
            Names of all objects registered in the tree
        """
        return [
            value["name"]
            for value in self.tree.values()
            if value["entity_type"] == "object"
        ]

    @property
    def list_data(self):
        """
        @property
        list_data

        Returns
        -------
        data: List[str]
            Names of all data registered in the tree
        """
        return [
            value["name"]
            for value in self.tree.values()
            if value["entity_type"] == "data"
        ]

    @property
    def root(self) -> "group.Group":
        """
        @property
        root

        Returns
        -------
        root: geoh5io.Group
            Group entity corresponding to the root (Workspace)
        """

        return self._root

    @property
    def h5file(self) -> Optional[str]:
        """
        @property
        h5file

        Returns
        -------
        h5file: str
            File name with path to the target geoh5 database
        """

        return self._h5file

    @h5file.setter
    def h5file(self, h5file):

        self._h5file = h5file

        if h5file is not None:

            # Check if the geoh5 exists, if not create it
            if not os.path.exists(self._h5file):
                H5Writer.create_geoh5(self)

    @staticmethod
    def active() -> Workspace:
        """ Get the active workspace. """
        active_one = Workspace._active_ref()
        if active_one is None:
            raise RuntimeError("No active workspace.")

        # so that type check does not complain of possible returned None
        return cast(Workspace, active_one)

    @staticmethod
    def save_entity(entity: Entity, close_file: bool = True):
        """
        save_entity(entity)

        Parameters
        ----------
        entity: geoh5io.Entity
            Entity to be written to the project geoh5 file
        close_file: bool optional
            Close the geoh5 database after writing is completed [True] or False

        """
        H5Writer.save_entity(entity, close_file=close_file)

    def finalize(self):
        """ Finalize the geoh5 file by re-building the Root"""
        H5Writer.finalize(self)

    def add_to_tree(
        self,
        entity: Entity,
        attributes: Optional[dict] = None,
        parent: Optional[uuid.UUID] = None,
        children: Optional[list] = None,
    ):
        """
        add_to_tree(entity)

        Add entity and attribute to tree for fast reference and write to geoh5

        Parameters
        ----------
        entity: geoh5io.Entity
            Entity to be added
        attributes: dict optional
            Dictionary of attributes to be written with the object
        parent: uuid.UUID optional
            Unique identifier of the parent Entity
        children: List[uuid] optional
            List of unique identifier of children entities
        """

        uid = entity.uid

        self.tree[uid] = {}

        if isinstance(entity, Group):
            entity_type = "group"
        elif isinstance(entity, Data):
            entity_type = "data"
        else:
            entity_type = "object"

        self.tree[uid]["entity_type"] = entity_type

        if attributes is not None:
            for key, value in attributes.items():
                self.tree[uid][key.replace(" ", "_").lower()] = value
        else:

            for key, attr in entity.__dict__.items():
                if key.replace("_", "") in ["workspace", "values"]:
                    continue
                if "uid" in key:
                    self.tree[uid]["id"] = H5Writer.uuid_value(attr)
                else:
                    if ("association" in key) and isinstance(
                        attr, data.DataAssociationEnum
                    ):
                        attr = attr.name.lower().capitalize()

                    self.tree[uid][
                        key.replace("_", " ").strip().replace(" ", "_")
                    ] = attr

        self.tree[uid]["type"] = entity.entity_type.uid

        if parent is None:
            self.tree[uid]["parent"] = []
        else:
            self.tree[uid]["parent"] = parent

        if children is None:
            self.tree[uid]["children"] = []
        else:
            self.tree[uid]["children"] = children

        # Store a pointer to the object in memory
        # self.tree[uid]["object"] = entity

        # Add type to tree
        self.tree[entity.entity_type.uid] = {}
        self.tree[entity.entity_type.uid]["entity_type"] = entity_type + "_type"

        for key, attr in entity.entity_type.__dict__.items():
            if key.replace("_", "") in ["workspace", "values"]:
                continue
            if "uid" in key:
                self.tree[entity.entity_type.uid]["id"] = H5Writer.uuid_value(attr)

            else:
                if ("primitive_type" in key) and isinstance(
                    attr, data.PrimitiveTypeEnum
                ):
                    attr = attr.name.lower().capitalize()

                self.tree[entity.entity_type.uid][
                    key.replace("_", " ").strip().replace(" ", "_")
                ] = attr

    def get_entity(self, name: str) -> List[Entity]:
        """
        get_entity(name)

        Retrieve an entity from one of its identifier, either by name or uuid

        Parameters
        ----------
        name: str | uuid.UUID
            Object identifier, either name or uuid

        Returns
        -------
        object_list: List[Entity]
            List of entities with the same given name
        """

        base_classes = {"group": Group, "object": ObjectBase, "data": Data}

        if isinstance(name, uuid.UUID):
            list_entity_uid = [name]

        else:  # Extract all objects uuid with matching name
            list_entity_uid = [
                key
                for key in self.tree
                if ("type" not in self.tree[key]["entity_type"])
                and (self.tree[key]["name"] == name)
            ]

        entity_list: List[Entity] = []
        for uid in list_entity_uid:
            entity_type = self.tree[uid]["entity_type"]

            # Check if an object already exists in the workspace
            finder = getattr(self, f"find_{entity_type}")
            if finder(uid) is not None:
                entity_list += [finder(uid)]
                continue

            created_object = self.create_entity(
                base_classes[entity_type],
                self.tree[uid]["name"],
                uid=uid,
                entity_type_uid=self.tree[uid]["type"],
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
        self, entity_class, name: str, uid: Optional[uuid.UUID] = uuid.uuid4(), **kwargs
    ):
        """
        create_entity(entity_class, name, uuid, type_uuid)

        Function to create and register a new entity and its entity_type.

        Parameters
        ----------
        entity_class: Entity
            Type of entity to be created
        name: str
            Name of the entity displayed in the project tree
        uid: uuid.UUID
            Unique identifier of the entity

        Returns
        -------
        entity: Entity
            New entity created
        """

        created_entity: Optional[Entity] = None

        if "entity_type_uid" in kwargs:
            entity_type_uid = kwargs["entity_type_uid"]
        # Assume that entity is being created from its class
        elif hasattr(entity_class, "default_type_uid"):
            entity_type_uid = entity_class.default_type_uid()
            entity_class = entity_class.__bases__
        else:
            raise RuntimeError(
                f"An entity_type_uid must be provided for {entity_class}."
            )

        if entity_class is not Data:
            for _, member in inspect.getmembers(groups) + inspect.getmembers(objects):

                if (
                    inspect.isclass(member)
                    and issubclass(member, entity_class)
                    and member is not entity_class
                    and hasattr(member, "default_type_uid")
                    and not member == CustomGroup
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
            attributes = kwargs["attributes"]
            type_attributes = kwargs["type_attributes"]

            data_type = data.data_type.DataType(
                self,
                entity_type_uid,
                getattr(
                    data.primitive_type_enum.PrimitiveTypeEnum,
                    type_attributes["primitive_type"].upper(),
                ),
            )
            data_type.name = name

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

        if isinstance(created_entity, Entity):
            # Add the new entity and type to tree
            if created_entity.uid not in self.tree.keys():
                self.add_to_tree(created_entity)

            if "parent" in kwargs:
                created_entity.parent = kwargs["parent"]
                # self.set_parent(kwargs['parent'])
            elif name != "Workspace":
                created_entity.parent = self.get_entity("Workspace")

            if isinstance(created_entity, Points):
                if "vertices" in kwargs:
                    created_entity.vertices = kwargs["vertices"]

                if "data" in kwargs:
                    data_objects = created_entity.add_data(kwargs["data"])

                    return tuple([created_entity] + data_objects)

        return created_entity

    def get_names_of_type(self, uid: uuid.UUID, children_type: str):
        """
        get_names_of_type(uid, children_type)

        Get all children's name of a certain type

        Parameters
        ----------
        uid: uuid.UUID
            Unique identifier of parent entity
        children_type: str
            Specify the type of children requested: "object", "group", "data"

        Returns
        -------
        children: list
            List of names of given type
        """

        name_list: List[str] = []

        for key in self.tree[uid]["children"]:

            if self.tree[key]["entity_type"] == children_type:
                name_list += [self.tree[key]["name"]]

        return name_list

    def get_child(self, parent_uuid: uuid.UUID, child_name: str) -> List[Entity]:
        """
        get_child(parent_uuid, child_name)

        Return a list of children entities with given name

        Parameters
        ----------
        parent_uuid: UUID
            The unique identifier of the parent entity
        child_name: str
            Name of entity(ies)

        Returns
        -------
        entity_list List[Entity]:
            A list of registered entities
        """

        if isinstance(child_name, uuid.UUID):
            return self.get_entity(child_name)

        for key in self.tree[parent_uuid]["children"]:

            if self.tree[key]["name"] == child_name:

                return self.get_entity(key)

        return []

    def get_parent(self, uid: uuid.UUID) -> List[Entity]:
        """
        get_parent(uid)

        Get the parent unique identifier of a given entity uuid

        Parameters
        ----------
        uid: uuid.UUID
            Unique identifier of child

        Returns
        -------
        parent: geoh5io.Entity
            The registered parent entity
        """

        if self.tree[uid]["parent"]:
            return self.get_entity(self.tree[uid]["parent"])

        return []

    def fetch_values(self, uid: uuid.UUID) -> Optional[float]:
        """
        fetch_values(uid)

        Fetch the data values from the source h5 file

        Parameters
        ----------
        uid: uuid.UUID
            Unique identifier of target data object

        Returns
        -------
        value: numpy.array
            Array of values
        """

        return H5Reader.fetch_values(self._h5file, self._base, uid)

    def fetch_vertices(self, uid: uuid.UUID) -> Coord3D:
        """
        Get the vertices of an object from the source h5 file

        Parameters
        ----------
        uid: uuid.UUID
            Unique identifier of target entity

        Returns
        -------
        coordinates: Coord3D
            Coordinate entity with locations
        """

        return H5Reader.fetch_vertices(self._h5file, self._base, uid)

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

    def get_workspace_attributes(self):
        """ Fetch the workspace attributes
        """

        if getattr(self, "_workspace_attributes", None) is None:

            self._workspace_attributes = {}

            for attr in dir(WorkspaceAttributes()):

                if "__" not in attr:
                    self._workspace_attributes[attr] = getattr(
                        WorkspaceAttributes(), attr
                    )

        return self._workspace_attributes


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
