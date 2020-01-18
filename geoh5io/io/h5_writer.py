import uuid
from typing import Optional

import h5py
from numpy import asarray, dtype, float64, int8

from geoh5io.data import Data, DataType
from geoh5io.groups import Group, GroupType
from geoh5io.objects import ObjectBase, ObjectType, Octree
from geoh5io.shared import Entity, EntityType
from geoh5io.workspace import RootGroup


class H5Writer:

    str_type = h5py.special_dtype(vlen=str)

    @staticmethod
    def bool_value(value: bool) -> int8:
        return int8(1 if value else 0)

    @staticmethod
    def uuid_value(value: uuid.UUID) -> str:
        """Return the string representation '{uuid}' used by the geoh5 database"""
        return f"{{{value}}}"

    @classmethod
    def create_geoh5(cls, workspace, file: str = None, close_file: bool = True):
        """
        create_geoh5(workspace, file=None, close_file=True)

        Create a geoh5 file and add the default groups structure

        Parameters
        ----------
        workspace: geoh5io.Workspace
            Workspace object defining the project structure

        file: str optional
            File name for the geoh5. Takes the default Workspace.h5file name if omitted

        close_file: bool optional
           Close h5 file after write [True] or False

        Returns
        -------
        h5file: h5py.File
            Pointer to a geoh5 file. Active link if "close_file" is False)
        """
        attr = workspace.get_workspace_attributes()

        # Take default name
        if file is None:
            file = workspace.h5file

        # Returns default error if already exists
        h5file = h5py.File(file, "w")

        # Write the workspace group
        project = h5file.create_group(workspace.base_name)
        project.attrs.create("Distance unit", attr["distance_unit"], dtype=cls.str_type)
        project.attrs.create("Version", attr["version"], dtype=type(attr["version"]))
        project.attrs.create("Contributors", attr["contributors"], dtype=cls.str_type)
        project.attrs.create("GA Version", attr["ga_version"], dtype=cls.str_type)

        # Create base entity structure for geoh5
        project.create_group("Data")
        project.create_group("Groups")
        project.create_group("Objects")
        types = project.create_group("Types")
        types.create_group("Data types")
        types.create_group("Group types")
        types.create_group("Object types")

        workspace_entity = workspace.get_entity("Workspace")[0]
        # Check if the Workspace already has a tree
        # if workspace.tree:
        #     workspace_entity = workspace.get_entity("Workspace")[0]
        #
        #     assert workspace_entity is not None, "The tree has no 'Workspace' group."
        #
        # else:
        #     workspace_entity = workspace.create_entity(
        #         Group,
        #         "Workspace",
        #         uuid.uuid4(),
        #         entity_type_uid=NoTypeGroup.default_type_uid(),
        #     )

        # workspace.add_to_tree(workspace_entity)

        workspace_handle = H5Writer.add_entity(file, workspace_entity, close_file=False)

        project["Root"] = workspace_handle

        if close_file:
            h5file.close()

        return h5file

    @classmethod
    def add_type(cls, file: str, entity_type: EntityType, close_file=True):
        """
        add_type(file, entity_type, close_file=True)

        Add a type to the geoh5 project.

        Parameters
        ----------
        file: str or h5py.File
            Name or handle to a geoh5 file

        entity_type: geoh5io.EntityType
            Entity_type to be added to the geoh5 file

        close_file: bool optional
           Close h5 file after write [True] or False

        Returns
        -------
        type: h5py.File
            Pointer to type in a geoh5 file. Active link if "close_file" is False
        """
        h5file = get_h5_handle(file)

        base = list(h5file.keys())[0]

        # tree = entity_type.workspace.tree
        uid = entity_type.uid

        # entity_type_str = tree[uid]["entity_type"].replace("_", " ").capitalize() + "s"
        if isinstance(entity_type, DataType):
            entity_type_str = "Data types"
        elif isinstance(entity_type, ObjectType):
            entity_type_str = "Object types"
        elif isinstance(entity_type, GroupType):
            entity_type_str = "Group types"
        else:
            return None

        # Check if already in the project
        if cls.uuid_value(uid) in list(h5file[base]["Types"][entity_type_str].keys()):

            if entity_type.update_h5:
                # Remove the entity type for re-write
                del h5file[base][entity_type][cls.uuid_value(uid)]
                entity_type.update_h5 = False
                entity_type.existing_h5_entity = False

            else:

                entity_type.existing_h5_entity = True
                return h5file[base]["Types"][entity_type_str][cls.uuid_value(uid)]

        new_type = h5file[base]["Types"][entity_type_str].create_group(
            cls.uuid_value(uid)
        )

        for key, value in entity_type.__dict__.items():
            if (
                key.replace("_", " ").strip().lower()
                in [
                    "uid",
                    "name",
                    "units",
                    "description",
                    "hidden",
                    "mapping",
                    "number of bins",
                    "transparent no data",
                    "datatype  primitive type",
                ]
                and value is not None
            ):

                if "uid" in key:
                    entry_key = "ID"
                    value = "{" + str(value) + "}"
                elif "primitive" in key:
                    entry_key = "Primitive type"
                    value = getattr(entity_type, key).name.lower().capitalize()
                else:
                    entry_key = key.replace("_", " ").strip().capitalize()

                if isinstance(value, (int8, bool)):
                    new_type.attrs.create(entry_key, int(value), dtype="int8")

                elif isinstance(value, str):
                    new_type.attrs.create(entry_key, value, dtype=cls.str_type)

                else:
                    new_type.attrs.create(entry_key, value, dtype=asarray(value).dtype)

        if close_file:
            h5file.close()

        entity_type.update_h5 = False
        entity_type.existing_h5_entity = True

        return new_type

    @classmethod
    def finalize(cls, workspace, file: str = None, close_file=False):
        """
        finalize(workspace, file=None, close_file=True)

        Build the Root of a project

        Parameters
        ----------
        workspace: geoh5io.Workspace
            Workspace object defining the project structure

        file: str optional
            File name for the geoh5. Takes the default Workspace.h5file name if omitted

        close_file: bool optional
           Close h5 file after write [True] or False
        """
        if file is None:
            h5file = h5py.File(workspace.h5file, "r+")

        else:
            if not isinstance(file, h5py.File):
                h5file = h5py.File(file, "r+")
            else:
                h5file = file

        workspace_group: Entity = workspace.get_entity("Workspace")[0]
        root_handle = H5Writer.fetch_handle(h5file, workspace_group)

        del h5file[workspace.base_name]["Root"]
        h5file[workspace.base_name]["Root"] = root_handle

        # # Refresh the project tree
        # workspace.tree = H5Reader.get_project_tree(
        #     workspace.h5file, workspace.base_name
        # )
        if close_file:
            h5file.close()

    @classmethod
    def add_vertices(cls, file: str, entity, close_file=True):
        """
        add_vertices(file, entity, close_file=True)

        Add vertices to a points object.

        Parameters
        ----------
        file: str or h5py.File
            Name or handle to a geoh5 file

        entity: geoh5io.Entity
            Target entity to which vertices are being written

        close_file: bool optional
           Close h5 file after write [True] or False
        """
        if not isinstance(file, h5py.File):
            h5file = h5py.File(file, "r+")
        else:
            h5file = file

        if hasattr(entity, "vertices") and entity.vertices:
            xyz = entity.vertices.locations
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            # Adding vertices
            loc_type = dtype([("x", float64), ("y", float64), ("z", float64)])

            vertices = entity_handle.create_dataset(
                "Vertices", (xyz.shape[0],), dtype=loc_type
            )
            vertices["x"] = xyz[:, 0]
            vertices["y"] = xyz[:, 1]
            vertices["z"] = xyz[:, 2]

        if close_file:
            h5file.close()

    @classmethod
    def add_cell_delimiters(cls, file: str, entity, close_file=True):
        """
        add_cell_delimiters(file, entity, close_file=True)

        Add (u, v, z) cell delimiters to a block_model object.

        Parameters
        ----------
        file: str or h5py.File
            Name or handle to a *.geoh5 file

        entity: geoh5io.Entity
            Target entity to which cells are being written

        close_file: bool optional
           Close h5 file after write [True] or False
        """
        if not isinstance(file, h5py.File):
            h5file = h5py.File(file, "r+")
        else:
            h5file = file

        if hasattr(entity, "u_cell_delimiters") and (
            entity.u_cell_delimiters is not None
        ):
            u_cell_delimiters = entity.u_cell_delimiters
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            # Adding cells
            entity_handle.create_dataset(
                "U cell delimiters",
                data=u_cell_delimiters,
                dtype=u_cell_delimiters.dtype,
            )

        if hasattr(entity, "v_cell_delimiters") and (
            entity.v_cell_delimiters is not None
        ):
            v_cell_delimiters = entity.v_cell_delimiters
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            # Adding cells
            entity_handle.create_dataset(
                "V cell delimiters",
                data=v_cell_delimiters,
                dtype=v_cell_delimiters.dtype,
            )

        if hasattr(entity, "z_cell_delimiters") and (
            entity.z_cell_delimiters is not None
        ):
            z_cell_delimiters = entity.z_cell_delimiters
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            # Adding cells
            entity_handle.create_dataset(
                "Z cell delimiters",
                data=z_cell_delimiters,
                dtype=z_cell_delimiters.dtype,
            )

        if close_file:
            h5file.close()

    @classmethod
    def add_cells(cls, file: str, entity, close_file=True):
        """
        add_cells(file, entity, close_file=True)

        Add cells to an object.

        Parameters
        ----------
        file: str or h5py.File
            Name or handle to a *.geoh5 file

        entity: geoh5io.Entity
            Target entity to which cells are being written

        close_file: bool optional
           Close h5 file after write [True] or False
        """
        if not isinstance(file, h5py.File):
            h5file = h5py.File(file, "r+")
        else:
            h5file = file

        if hasattr(entity, "cells") and (entity.cells is not None):
            indices = entity.cells
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            # Adding cells
            entity_handle.create_dataset(
                "Cells", indices.shape, data=indices, dtype=indices.dtype
            )

        if close_file:
            h5file.close()

    @classmethod
    def add_octree_cells(cls, file: str, entity, close_file=True):
        """
        add_octree_cells(file, entity, close_file=True)

        Add octree cells to an object.

        Parameters
        ----------
        file: str or h5py.File
            Name or handle to a *.geoh5 file

        entity: geoh5io.Entity
            Target entity to which cells are being written

        close_file: bool optional
           Close h5 file after write [True] or False
        """
        if not isinstance(file, h5py.File):
            h5file = h5py.File(file, "r+")
        else:
            h5file = file

        if hasattr(entity, "octree_cells") and (entity.octree_cells is not None):
            octree_cells = entity.octree_cells
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            # Adding cells
            entity_handle.create_dataset(
                "Octree Cells",
                octree_cells.shape,
                data=octree_cells,
                dtype=octree_cells.dtype,
            )

        if close_file:
            h5file.close()

    @classmethod
    def add_data_values(cls, file: str, entity, values, close_file=True):
        """
        add_data_values(file, entity, values=None, close_file=True)

        Add a type to the geoh5 project.

        Parameters
        ----------
        file: str or h5py.File
            Name or handle to a geoh5 file

        entity: geoh5io.Entity
            Target entity to which vertices are being written

        values: numpy.array or str
            Array of values to be added to the geoh5 file

        close_file: bool optional
           Close h5 file after write [True] or False
        """
        if not isinstance(file, h5py.File):
            h5file = h5py.File(file, "r+")
        else:
            h5file = file

        entity_handle = H5Writer.fetch_handle(h5file, entity)

        # Adding an array of values
        entity_handle.create_dataset("Data", data=values)

        if close_file:
            h5file.close()

    @classmethod
    def fetch_handle(cls, file: str, entity):
        """
        fetch_handle(file, entity)

        Parameters
        ----------
        file: str
            Target geoh5 file

        entity: Entity
            Entity to be retrieved from the geoh5 file

        Returns
        -------
        entity_handle: h5py.File
            Pointer to an existing entity or None if not found
        """
        cls.str_type = h5py.special_dtype(vlen=str)

        h5file = get_h5_handle(file)

        base = list(h5file.keys())[0]

        # tree = entity.workspace.tree
        uid = entity.uid

        # entity_type = tree[uid]["entity_type"].capitalize()

        if isinstance(entity, Data):
            entity_type = "Data"
        elif isinstance(entity, ObjectBase):
            entity_type = "Objects"
        elif isinstance(entity, Group):
            entity_type = "Groups"
        else:
            return None

        # if entity_type != "Data":
        #     entity_type += "s"

        # Check if already in the project
        if cls.uuid_value(uid) in list(h5file[base][entity_type].keys()):

            return h5file[base][entity_type][cls.uuid_value(uid)]

        return None

    @classmethod
    def add_entity(cls, file: str, entity, values=None, close_file=True):
        """
        add_entity(file, entity_type, values=None, close_file=True)

        Add an entity, its attributes and values to a geoh5 project.
        If the entity is already in the geoh5, the function returns a
        pointer to the object on file.

        Parameters
        ----------
        file: str or h5py.File
            Name or handle to a geoh5 file

        entity: geoh5io.Entity
            Entity to be added to the geoh5 file

        values: numpy.array optional
            Array of values to be added to Data entity

        close_file: bool optional = True
           Close h5 file after write

        Returns
        -------
        type: h5py.File
            Pointer to written entity. Active link if "close_file" is False
        """
        cls.str_type = h5py.special_dtype(vlen=str)

        h5file = get_h5_handle(file)

        base = list(h5file.keys())[0]

        if isinstance(entity, Data):
            entity_type = "Data"
        elif isinstance(entity, ObjectBase):
            entity_type = "Objects"
        elif isinstance(entity, Group):
            entity_type = "Groups"
        else:
            return None

        uid = entity.uid

        # # entity_type = tree[uid]["entity_type"].capitalize()
        #
        # if entity_type != "Data":
        #     entity_type += "s"

        # Check if already in the project
        if cls.uuid_value(uid) in list(h5file[base][entity_type].keys()):

            if entity.update_h5:
                # Remove the entity for re-write
                del h5file[base][entity_type][cls.uuid_value(uid)]
                entity.update_h5 = False
                entity.existing_h5_entity = False

            else:
                # Check if file reference to a hdf5
                if close_file:
                    h5file.close()
                    return None
                entity.existing_h5_entity = True
                return h5file[base][entity_type][cls.uuid_value(uid)]

        entity_handle = h5file[base][entity_type].create_group(cls.uuid_value(uid))

        if entity_type == "Groups":
            entity_handle.create_group("Data")
            entity_handle.create_group("Groups")
            entity_handle.create_group("Objects")
        elif entity_type == "Objects":
            entity_handle.create_group("Data")

        for key, value in entity.__dict__.items():
            if key.replace("_", " ").strip().lower() in [
                "allow delete",
                "allow move",
                "allow rename",
                "dip",
                "uid",
                "last focus",
                "name",
                "origin",
                "public",
                "rotation",
                "u count",
                "u size",
                "u cell size",
                "v count",
                "v size",
                "v cell size",
                "w count",
                "w cell size",
                "vertical",
                "association",
            ]:

                if "uid" in key:
                    entry_key = "ID"
                    value = "{" + str(value) + "}"
                else:
                    entry_key = key.replace("_", " ").strip().capitalize()

                    if ("count" in key) and isinstance(entity, Octree):
                        entry_key = "N" + entry_key.replace(" count", "")

                    if entry_key == "Association":
                        value = value.name.capitalize()

                # More custom upper/lower
                entry_key = entry_key.replace(" size", " Size")
                entry_key = entry_key.replace(" count", " Count")
                entry_key = entry_key.replace(" cell", " Cell")

                if isinstance(value, (int8, bool)):
                    entity_handle.attrs.create(entry_key, int(value), dtype="int8")

                elif isinstance(value, str):
                    entity_handle.attrs.create(entry_key, value, dtype=cls.str_type)

                else:
                    entity_handle.attrs.create(
                        entry_key, value, dtype=asarray(value).dtype
                    )

        # Add the type and return a pointer
        new_type = H5Writer.add_type(h5file, entity.entity_type, close_file=False)

        entity_handle["Type"] = new_type

        cls.add_attributes(h5file, entity, values=values)

        # Check if file reference to a hdf5
        if close_file:
            h5file.close()

        entity.update_h5 = False
        entity.existing_h5_entity = True

        return entity_handle

    @classmethod
    def add_to_parent(
        cls,
        file: str,
        child_entity: Entity,
        parent=None,
        close_file=True,
        recursively=False,
    ):
        """
        add_to_parent(file, child_entity, close_file=True, recursively=False)

        Add an entity, its attributes and values to a geoh5 project.
        If the entity is already in the geoh5, the function returns a
        pointer to the object on file.

        Parameters
        ----------
        file: str or h5py.File
            Name or handle to a geoh5 file

        child_entity: geoh5io.Entity
            Entity to be added or linked to a parent in geoh5

        parent: geoh5io.Entity
            Parent entity to be written under Entity or [None]

        close_file: bool optional
           Close h5 file after write: [True] or False

        recursively: bool optional = False
            Add parents recursively until reaching the top Workspace group: True or [False]
        """

        h5file = get_h5_handle(file)

        # If RootGroup than no parent to be added
        if isinstance(child_entity, RootGroup):
            return

        cls.str_type = h5py.special_dtype(vlen=str)

        # Check if changing workspace
        if parent is None:
            parent = child_entity.parent

        uid = child_entity.uid

        # Get the h5 handle
        child_entity_handle = H5Writer.add_entity(
            h5file, child_entity, close_file=False
        )

        parent_handle = H5Writer.add_entity(h5file, parent, close_file=False)

        if isinstance(child_entity, Data):
            entity_type = "Data"
        elif isinstance(child_entity, ObjectBase):
            entity_type = "Objects"
        elif isinstance(child_entity, Group):
            entity_type = "Groups"
        else:
            if close_file:
                h5file.close()
            return

        # Check if child h5py.Group already exists
        if entity_type not in parent_handle.keys():
            parent_handle.create_group(entity_type)

        # Check if child uuid not already in h5
        if cls.uuid_value(uid) not in list(parent_handle[entity_type].keys()):
            parent_handle[entity_type][cls.uuid_value(uid)] = child_entity_handle

        if recursively:
            H5Writer.add_to_parent(
                h5file, parent, close_file=False, recursively=recursively
            )

        # Close file if requested
        if close_file:
            h5file.close()

    @classmethod
    def save_entity(
        cls,
        entity: Entity,
        parent: Optional[Entity] = None,
        file: str = None,
        close_file=True,
        add_children=True,
    ):
        """
        save_entity(entity, file, close_file=True)

        Function to add an entity to geoh5 with its parents

        Parameters
        ----------
        entity: geoh5io.Entity
            Entity to be added to a geoh5

        parent: geoh5io.Entity = None
            Parent entity to be written under

        file: str or h5py.File
            Name or handle to a geoh5 file

        close_file: bool optional = True
           Close h5 file after write

        add_children: bool optional = True
            Add children associated with entity

        """

        cls.str_type = h5py.special_dtype(vlen=str)

        if file is not None:
            # Check if file reference to a hdf5
            if not isinstance(file, h5py.File):
                h5file = h5py.File(file, "r+")
            else:
                h5file = file
        else:
            h5file = h5py.File(entity.workspace.h5file, "r+")

        # Add itself to the project
        new_entity = H5Writer.add_entity(h5file, entity, close_file=False)

        if add_children:
            # Write children entities and add to current parent
            for child in entity.children:
                H5Writer.add_entity(h5file, child, close_file=False)
                H5Writer.add_to_parent(
                    h5file, child, close_file=False, recursively=False
                )

        H5Writer.add_to_parent(
            h5file, entity, parent=parent, close_file=False, recursively=True
        )

        # Check if file reference to a hdf5
        if close_file:
            h5file.close()

        return new_entity

    @staticmethod
    def add_attributes(h5file, entity, values=None):
        """

        Parameters
        ----------
        h5file: str or h5py.File
            Name or handle to a geoh5 file

        entity: geoh5io.Entity
            Entity to be added to the geoh5 file

        values: numpy.array optional
            Array of values to be added to Data entity

        """
        if hasattr(entity, "values"):
            if values is not None:
                H5Writer.add_data_values(
                    h5file, entity, values=values, close_file=False
                )

            if entity.values is not None:
                H5Writer.add_data_values(
                    h5file, entity, values=entity.values, close_file=False
                )

        if hasattr(entity, "vertices") and entity.vertices:
            H5Writer.add_vertices(h5file, entity, close_file=False)

        if (
            hasattr(entity, "u_cell_delimiters")
            and entity.u_cell_delimiters is not None
        ):
            H5Writer.add_cell_delimiters(h5file, entity, close_file=False)

        if hasattr(entity, "cells") and entity.cells is not None:
            H5Writer.add_cells(h5file, entity, close_file=False)

        if hasattr(entity, "octree_cells") and entity.octree_cells is not None:
            H5Writer.add_octree_cells(h5file, entity, close_file=False)


def get_h5_handle(file):
    """
    get_h5_handle(file)

    Check if file reference to an existing hdf5

    Returns
    -------
    h5py: h5py.File
        Handle to an opened h5py file

    """

    if not isinstance(file, h5py.File):
        h5file = h5py.File(file, "r+")
    else:
        h5file = file

    return h5file
