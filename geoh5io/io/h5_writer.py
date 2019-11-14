import uuid

import h5py
import numpy as np

from geoh5io.groups import Group, NoTypeGroup
from geoh5io.shared import Entity, EntityType


class H5Writer:

    str_type = h5py.special_dtype(vlen=str)

    @staticmethod
    def bool_value(value: bool) -> np.uint8:
        return np.uint8(1 if value else 0)

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
           Close h5 file after write [True] | False

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

        # Check if the Workspace already has a tree
        if workspace.tree:
            workspace_entity = workspace.get_entity("Workspace")[0]

            assert workspace_entity is not None, "The tree has no 'Workspace' group."

        else:
            workspace_entity = workspace.create_entity(
                Group, "Workspace", uuid.uuid4(), NoTypeGroup.default_type_uid()
            )

            workspace.add_to_tree(workspace_entity)

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
        file: str | h5py.File
            Name or handle to a geoh5 file

        entity_type: geoh5io.EntityType
            Entity_type to be added to the geoh5 file

        close_file: bool optional
           Close h5 file after write [True] | False

        Returns
        -------
        type: h5py.File
            Pointer to type in a geoh5 file. Active link if "close_file" is False
        """
        # Check if file reference to a hdf5
        if not isinstance(file, h5py.File):
            h5file = h5py.File(file, "r+")
        else:
            h5file = file

        base = list(h5file.keys())[0]

        tree = entity_type.workspace.tree
        uid = entity_type.uid

        entity_type_str = tree[uid]["entity_type"].replace("_", " ").capitalize() + "s"
        new_type = h5file[base]["Types"][entity_type_str].create_group(
            cls.uuid_value(uid)
        )

        for key, value in tree[uid].items():
            if (key == "entity_type") or ("allow" in key):
                continue

            if key == "id":
                entry_key = key.replace("_", " ").upper()
            else:
                entry_key = key.replace("_", " ").capitalize()

            if isinstance(value, str):
                new_type.attrs.create(entry_key, value, dtype=cls.str_type)
            else:
                if value is not None:
                    new_type.attrs.create(entry_key, value, dtype=type(value))

        if close_file:
            h5file.close()

        return new_type

    @classmethod
    def finalize(cls, workspace, file: str = None, close_file=True):
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
           Close h5 file after write [True] | False
        """

        if file is None:
            h5file = h5py.File(workspace.h5file, "r+")

        else:
            if not isinstance(file, h5py.File):
                h5file = h5py.File(file, "r+")
            else:
                h5file = file

        workspace_group: Entity = workspace.get_entity("Workspace")[0]
        H5Writer.add_entity(h5file, workspace_group, close_file=False)

        if close_file:
            h5file.close()

    @classmethod
    def add_vertices(cls, file: str, entity, close_file=True):
        """
        add_vertices(file, entity, close_file=True)

        Add a type to the geoh5 project.

        Parameters
        ----------
        file: str | h5py.File
            Name or handle to a geoh5 file

        entity: geoh5io.Entity
            Target entity to which vertices are being written

        close_file: bool optional
           Close h5 file after write [True] | False
        """

        if not isinstance(file, h5py.File):
            h5file = h5py.File(file, "r+")
        else:
            h5file = file

        if hasattr(entity, "vertices") and entity.vertices:
            xyz = entity.vertices.locations
            entity_handle = H5Writer.add_entity(h5file, entity, close_file=False)

            # Adding vertices
            loc_type = np.dtype(
                [("x", np.float64), ("y", np.float64), ("z", np.float64)]
            )

            vertices = entity_handle.create_dataset(
                "Vertices", (xyz.shape[0],), dtype=loc_type
            )
            vertices["x"] = xyz[:, 0]
            vertices["y"] = xyz[:, 1]
            vertices["z"] = xyz[:, 2]

        if close_file:
            h5file.close()

    @classmethod
    def add_data_values(cls, file: str, entity, values, close_file=True):
        """
        add_data_values(file, entity, values=None, close_file=True)

        Add a type to the geoh5 project.

        Parameters
        ----------
        file: str | h5py.File
            Name or handle to a geoh5 file

        entity: geoh5io.Entity
            Target entity to which vertices are being written

        values: numpy.array
            Array of values to be added to the geoh5 file

        close_file: bool optional
           Close h5 file after write [True] | False
        """

        if not isinstance(file, h5py.File):
            h5file = h5py.File(file, "r+")
        else:
            h5file = file

        entity_handle = H5Writer.add_entity(h5file, entity, close_file=False)

        # Adding an array of values
        entity_handle.create_dataset("Data", data=values)

        if close_file:
            h5file.close()

    @classmethod
    def add_entity(cls, file: str, entity, values=None, close_file=True):
        """
        add_entity(file, entity_type, values=None, close_file=True)

        Add an entity, its attributes and values to a geoh5 project.
        If the entity is already in the geoh5, the function returns a
        pointer to the object on file.

        Parameters
        ----------
        file: str | h5py.File
            Name or handle to a geoh5 file

        entity: geoh5io.Entity
            Entity to be added to the geoh5 file

        values: numpy.array optional
            Array of values to be added to Data entity

        close_file: bool optional
           Close h5 file after write [True] | False

        Returns
        -------
        type: h5py.File
            Pointer to written entity. Active link if "close_file" is False
        """

        cls.str_type = h5py.special_dtype(vlen=str)
        # Check if file reference to a hdf5
        if not isinstance(file, h5py.File):
            h5file = h5py.File(file, "r+")
        else:
            h5file = file

        base = list(h5file.keys())[0]

        tree = entity.entity_type.workspace.tree
        uid = entity.uid

        entity_type = tree[uid]["entity_type"].capitalize()

        if entity_type != "Data":
            entity_type += "s"

        # Check if already in the project
        if cls.uuid_value(uid) in list(h5file[base][entity_type].keys()):
            # Check if file reference to a hdf5
            if close_file:
                h5file.close()
                return None
            return h5file[base][entity_type][cls.uuid_value(uid)]

        entity_handle = h5file[base][entity_type].create_group(cls.uuid_value(uid))

        if entity_type == "Groups":
            entity_handle.create_group("Data")
            entity_handle.create_group("Groups")
            entity_handle.create_group("Objects")
        elif entity_type == "Objects":
            entity_handle.create_group("Data")

        for key, value in tree[uid].items():
            if key in [
                "type",
                "entity_type",
                "parent",
                "children",
                "vertices",
                "visible",
            ]:
                continue
            if key == "id":
                entry_key = key.replace("_", " ").upper()
            else:
                entry_key = key.replace("_", " ").capitalize()

            if isinstance(value, int):
                entity_handle.attrs.create(entry_key, value, dtype="int8")

            else:
                entity_handle.attrs.create(entry_key, value, dtype=cls.str_type)

        # Add the type and return a pointer
        new_type = H5Writer.add_type(h5file, entity.entity_type, close_file=False)

        entity_handle["Type"] = new_type

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

        # Check if file reference to a hdf5
        if close_file:
            h5file.close()

        return entity_handle

    @classmethod
    def add_to_parent(
        cls, file: str, child_entity: Entity, close_file=True, recursively=False
    ):
        """
        add_to_parent(file, child_entity, close_file=True, recursively=False)

        Add an entity, its attributes and values to a geoh5 project.
        If the entity is already in the geoh5, the function returns a
        pointer to the object on file.

        Parameters
        ----------
        file: str | h5py.File
            Name or handle to a geoh5 file

        child_entity: geoh5io.Entity
            Entity to be added or linked to a parent in geoh5

        close_file: bool optional
           Close h5 file after write: [True] | False

        recursively: bool optional
            Add parents recursively until reaching the top Workspace group: True | [False]
        """

        cls.str_type = h5py.special_dtype(vlen=str)

        workspace = child_entity.entity_type.workspace
        tree = workspace.tree
        uid = child_entity.uid

        if not tree[uid]["parent"]:
            return

        # Check if file reference to a hdf5
        if not isinstance(file, h5py.File):
            h5file = h5py.File(file, "r+")
        else:
            h5file = file

        # Get the h5 handle
        child_entity_handle = H5Writer.add_entity(
            h5file, child_entity, close_file=False
        )

        parent = workspace.get_entity(tree[uid]["parent"])[0]
        parent_handle = H5Writer.add_entity(h5file, parent, close_file=False)

        entity_type = tree[uid]["entity_type"].capitalize()

        if entity_type != "Data":
            entity_type += "s"

        if entity_type not in parent_handle.keys():
            parent_handle.create_group(entity_type)

        if cls.uuid_value(uid) not in list(parent_handle[entity_type].keys()):
            parent_handle[entity_type][cls.uuid_value(uid)] = child_entity_handle

        if recursively:
            H5Writer.add_to_parent(h5file, parent, close_file=False)

        # Close file if requested
        if close_file:
            h5file.close()

    @classmethod
    def save_entity(cls, entity: Entity, file: str = None, close_file=True):
        """
        save_entity(entity, file, close_file=True)

        Function to add an entity to geoh5 with its parents

        Parameters
        ----------
        entity: geoh5io.Entity
            Entity to be added to a geoh5

        file: str | h5py.File
            Name or handle to a geoh5 file

        close_file: bool optional
           Close h5 file after write: [True] | False

        """

        cls.str_type = h5py.special_dtype(vlen=str)

        if file is not None:
            # Check if file reference to a hdf5
            if not isinstance(file, h5py.File):
                h5file = h5py.File(file, "r+")
            else:
                h5file = file

        else:
            h5file = h5py.File(entity.entity_type.workspace.h5file, "r+")

        workspace = entity.entity_type.workspace
        tree = workspace.tree
        uid = entity.uid

        # Add itself to the project
        new_entity = H5Writer.add_entity(h5file, entity, close_file=False)

        # Write children entities and add to current parent
        if tree[uid]["children"]:
            for child in tree[uid]["children"]:
                child_entity = workspace.get_entity(child)[0]
                H5Writer.add_entity(h5file, child_entity, close_file=False)
                H5Writer.add_to_parent(
                    h5file, child_entity, close_file=False, recursively=False
                )

        H5Writer.add_to_parent(h5file, entity, close_file=False, recursively=True)

        # Check if file reference to a hdf5
        if close_file:
            h5file.close()

        return new_entity
