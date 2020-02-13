import uuid

import h5py
from numpy import asarray, dtype, float64, int8

from geoh5io.data import Data, DataType
from geoh5io.groups import Group, GroupType
from geoh5io.objects import ObjectBase, ObjectType
from geoh5io.shared import Entity
from geoh5io.workspace import RootGroup


class H5Writer:

    str_type = h5py.special_dtype(vlen=str)

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
        # attr = workspace.get_workspace_attributes()

        # Take default name
        if file is None:
            file = workspace.h5file

        # Returns default error if already exists
        h5file = h5py.File(file, "w-")

        # Write the workspace group
        project = h5file.create_group(workspace.name)

        cls.add_attributes(file, workspace, close_file=False)

        # Create base entity structure for geoh5
        project.create_group("Data")
        project.create_group("Groups")
        project.create_group("Objects")
        types = project.create_group("Types")
        types.create_group("Data types")
        types.create_group("Group types")
        types.create_group("Object types")

        if close_file:
            h5file.close()

        return h5file

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
        h5file = cls.fetch_h5_handle(file, entity)

        if getattr(entity, "vertices", None) is not None:
            xyz = entity.vertices
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
        h5file = cls.fetch_h5_handle(file, entity)

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
        h5file = cls.fetch_h5_handle(file, entity)

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
        h5file = cls.fetch_h5_handle(file, entity)

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
        add_data_values(file, entity, values, close_file=True)

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
        h5file = cls.fetch_h5_handle(file, entity)

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

        h5file = cls.fetch_h5_handle(file, entity)

        base = list(h5file.keys())[0]

        if entity.name == base:
            return h5file[base]

        uid = entity.uid

        if isinstance(entity, Data):
            base_handle = h5file[base]["Data"]
        elif isinstance(entity, ObjectBase):
            base_handle = h5file[base]["Objects"]
        elif isinstance(entity, Group):
            base_handle = h5file[base]["Groups"]
        elif isinstance(entity, DataType):
            base_handle = h5file[base]["Types"]["Data types"]
        elif isinstance(entity, ObjectType):
            base_handle = h5file[base]["Types"]["Object types"]
        elif isinstance(entity, GroupType):
            base_handle = h5file[base]["Types"]["Group types"]
        else:
            raise RuntimeError(f"Cannot add object '{entity}' to geoh5.")

        # Check if already in the project
        if cls.uuid_str(uid) in list(base_handle.keys()):
            return base_handle[cls.uuid_str(uid)]

        return None

    @classmethod
    def finalize(cls, workspace, close_file=False):
        """
        finalize(workspace, file=None, close_file=True)

        Build the Root of a project

        Parameters
        ----------
        workspace: geoh5io.Workspace
            Workspace object defining the project structure

        close_file: bool optional
           Close h5 file after write [True] or False
        """
        h5file = cls.fetch_h5_handle(workspace.h5file, workspace)
        workspace_group: Entity = workspace.get_entity("Workspace")[0]
        root_handle = H5Writer.fetch_handle(h5file, workspace_group)

        if "Root" in h5file[workspace.name].keys():
            del h5file[workspace.name]["Root"]

        h5file[workspace.name]["Root"] = root_handle

        if close_file:
            h5file.close()

    @classmethod
    def add_entity(cls, entity, file=None, values=None, close_file=True):
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

        h5file = cls.fetch_h5_handle(file, entity)

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

        # Check if already in the project
        if cls.uuid_str(uid) in list(h5file[base][entity_type].keys()):

            if any([entity.modified_attributes]):
                cls.update_attributes(file, entity, close_file=False)
                entity.modified_attributes = []
                entity.existing_h5_entity = True

            else:
                # Check if file reference to a hdf5
                if close_file:
                    h5file.close()
                    return None
                entity.existing_h5_entity = True
            return h5file[base][entity_type][cls.uuid_str(uid)]

        entity_handle = h5file[base][entity_type].create_group(cls.uuid_str(uid))

        if entity_type == "Groups":
            entity_handle.create_group("Data")
            entity_handle.create_group("Groups")
            entity_handle.create_group("Objects")
        elif entity_type == "Objects":
            entity_handle.create_group("Data")

        H5Writer.add_attributes(file, entity, close_file=False)

        # Add the type and return a pointer
        new_type = H5Writer.add_type(h5file, entity, close_file=False)
        entity_handle["Type"] = new_type

        entity.entity_type.modified_attributes = []
        entity.entity_type.existing_h5_entity = True

        cls.add_datasets(entity, file=h5file, values=values, close_file=False)

        # Check if file reference to a hdf5
        if close_file:
            h5file.close()

        entity.modified_attributes = []
        entity.existing_h5_entity = True

        return entity_handle

    @classmethod
    def add_type(cls, file: str, entity: Entity, close_file=True):
        """
        add_type(file, entity_type, close_file=True)

        Add a type to the geoh5 project.

        Parameters
        ----------
        file: str or h5py.File
            Name or handle to a geoh5 file

        entity: Entity with type to be added to the geoh5 file

        close_file: bool optional
           Close h5 file after write [True] or False

        Returns
        -------
        type: h5py.File
            Pointer to type in a geoh5 file. Active link if "close_file" is False
        """
        entity_type = entity.entity_type

        h5file = cls.fetch_h5_handle(file, entity_type)

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
        if cls.uuid_str(uid) in list(h5file[base]["Types"][entity_type_str].keys()):

            if any([entity_type.modified_attributes]):
                cls.update_attributes(file, entity_type, close_file=False)
                entity.modified_attributes = []
                entity_type.existing_h5_entity = False

            else:
                entity_type.existing_h5_entity = True
                return h5file[base]["Types"][entity_type_str][cls.uuid_str(uid)]

        new_type = h5file[base]["Types"][entity_type_str].create_group(
            cls.uuid_str(uid)
        )
        H5Writer.add_attributes(file, entity_type, close_file=False)

        if close_file:
            h5file.close()

        entity_type.modified_attributes = False
        entity_type.existing_h5_entity = True

        return new_type

    @classmethod
    def add_to_parent(
        cls, entity: Entity, file=None, close_file=True, recursively=False
    ):
        """
        add_to_parent(file, entity, close_file=True, recursively=False)

        Add an entity, its attributes and values to a geoh5 project.
        If the entity is already in the geoh5, the function returns a
        pointer to the object on file.

        Parameters
        ----------
        file: str or h5py.File
            Name or handle to a geoh5 file

        entity: geoh5io.Entity
            Entity to be added or linked to a parent in geoh5

        parent: geoh5io.Entity
            Parent entity to be written under Entity or [None]

        close_file: bool optional
           Close h5 file after write: [True] or False

        recursively: bool optional = False
            Add parents recursively until reaching the top Workspace group: True or [False]
        """

        h5file = cls.fetch_h5_handle(file, entity)

        # If RootGroup than no parent to be added
        if isinstance(entity, RootGroup):
            return

        # cls.str_type = h5py.special_dtype(vlen=str)

        uid = entity.uid

        # Get the h5 handle
        entity_handle = H5Writer.add_entity(entity, file=h5file, close_file=False)

        parent_handle = H5Writer.add_entity(
            entity.parent, file=h5file, close_file=False
        )

        if isinstance(entity, Data):
            entity_type = "Data"
        elif isinstance(entity, ObjectBase):
            entity_type = "Objects"
        elif isinstance(entity, Group):
            entity_type = "Groups"
        else:
            if close_file:
                h5file.close()
            return

        # Check if child h5py.Group already exists
        if entity_type not in parent_handle.keys():
            parent_handle.create_group(entity_type)

        # Check if child uuid not already in h5
        if cls.uuid_str(uid) not in list(parent_handle[entity_type].keys()):
            parent_handle[entity_type][cls.uuid_str(uid)] = entity_handle

        if recursively:
            H5Writer.add_to_parent(
                entity.parent, file=h5file, close_file=False, recursively=recursively
            )

        # Close file if requested
        if close_file:
            h5file.close()

    @classmethod
    def save_entity(
        cls, entity: Entity, file: str = None, close_file=True, add_children=True
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
        h5file = cls.fetch_h5_handle(file, entity)

        # Add itself to the project
        new_entity = H5Writer.add_entity(entity, file=h5file, close_file=False)

        if add_children:
            # Write children entities and add to current parent
            for child in entity.children:
                H5Writer.add_entity(child, file=h5file, close_file=False)
                H5Writer.add_to_parent(
                    child, file=h5file, close_file=False, recursively=False
                )

        H5Writer.add_to_parent(entity, file=h5file, close_file=False, recursively=True)

        # Check if file reference to a hdf5
        if close_file:
            h5file.close()

        return new_entity

    @classmethod
    def add_datasets(
        cls, entity: Entity, file: str = None, values=None, close_file=True
    ):
        """

        Parameters
        ----------
        entity: geoh5io.Entity
            Entity to be added to the geoh5 file

        file: str or h5py.File
            Name or handle to a geoh5 file

        values: numpy.array optional
            Array of values to be added to Data entity

        close_file: bool optional = True
           Close h5 file after write

        """
        h5file = cls.fetch_h5_handle(file, entity)

        if hasattr(entity, "values"):
            if values is not None:
                H5Writer.add_data_values(h5file, entity, values, close_file=False)

            if isinstance(entity, Data):
                H5Writer.add_data_values(
                    h5file, entity, entity.values, close_file=False
                )

        if isinstance(entity, ObjectBase) and isinstance(entity.property_groups, list):
            H5Writer.add_property_groups(h5file, entity, close_file=False)

        if getattr(entity, "vertices", None) is not None:
            H5Writer.add_vertices(h5file, entity, close_file=False)

        if getattr(entity, "u_cell_delimiters", None) is not None:
            H5Writer.add_cell_delimiters(h5file, entity, close_file=False)

        if getattr(entity, "cells", None) is not None:
            H5Writer.add_cells(h5file, entity, close_file=False)

        if getattr(entity, "octree_cells", None) is not None:
            H5Writer.add_octree_cells(h5file, entity, close_file=False)

        if close_file:
            h5file.close()

    @classmethod
    def add_attributes(cls, file: str, entity, close_file=True):
        """

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

        """
        h5file = cls.fetch_h5_handle(file, entity)
        entity_handle = H5Writer.fetch_handle(file, entity)
        str_type = h5py.special_dtype(vlen=str)

        for key, attr in entity.attribute_map.items():

            try:
                value = getattr(entity, attr)
            except AttributeError:
                continue

            if key == "ID":
                value = "{" + str(value) + "}"

            if key == "PropertyGroups":
                continue

            if key in ["Association", "Primitive type"]:
                value = value.name.lower().capitalize()

            if isinstance(value, (int8, bool)):
                entity_handle.attrs.create(key, int(value), dtype="int8")

            elif isinstance(value, str):
                entity_handle.attrs.create(key, value, dtype=str_type)

            elif value is None:
                entity_handle.attrs.create(key, "None", dtype=str_type)

            else:
                entity_handle.attrs.create(key, value, dtype=asarray(value).dtype)
        if close_file:
            h5file.close()

    @classmethod
    def add_property_groups(cls, h5file, entity, close_file=True):
        """
        add_property_groups(h5file, entity)

        Parameters
        ----------
        h5file: str or h5py.File
            Name or handle to a geoh5 file

        entity: geoh5io.Entity
            Entity to be added to the geoh5 file

        close_file: bool optional = True
           Close h5 file after write
        """
        if hasattr(entity, "property_groups") and isinstance(
            entity.property_groups, list
        ):

            entity_handle = H5Writer.fetch_handle(h5file, entity)

            # Check if a group already exists, then remove and write
            if "PropertyGroups" in entity_handle.keys():
                del entity_handle["PropertyGroups"]

            entity_handle.create_group("PropertyGroups")
            for p_g in entity.property_groups:

                uid = cls.uuid_str(p_g.uid)
                if uid in entity_handle["PropertyGroups"].keys():
                    del entity_handle["PropertyGroups"][uid]

                entity_handle["PropertyGroups"].create_group(uid)

                group_handle = entity_handle["PropertyGroups"][uid]

                for key, attr in p_g.attribute_map.items():

                    try:
                        value = getattr(p_g, attr)
                    except AttributeError:
                        continue

                    if key == "Association":
                        value = value.name.capitalize()

                    elif key == "Properties":
                        value = asarray([cls.uuid_str(val) for val in value])

                    elif key == "ID":
                        value = cls.uuid_str(value)

                    group_handle.attrs.create(
                        key, value, dtype=h5py.special_dtype(vlen=str)
                    )
        if close_file:
            h5file.close()

    @classmethod
    def update_attributes(cls, file: str, entity, close_file=True):
        """
        update_attributes(h5file, entity, close_file=True)

        Update the attributes of an entity specified by the h5

        Parameters
        ----------
        file: str or h5py.File
            Name or handle to a geoh5 file

        entity: geoh5io.Entity
            Entity to be added to the geoh5 file

        close_file: bool optional = True
           Close h5 file after write
        """
        entity_handle = H5Writer.fetch_handle(file, entity)

        for attr in entity.modified_attributes:
            if attr == "values":
                del entity_handle["Data"]
                cls.add_data_values(file, entity, entity.values, close_file=close_file)

            elif attr == "cells":
                del entity_handle["Cells"]
                cls.add_cells(file, entity, close_file=close_file)

            elif attr == "vertices":
                del entity_handle["Vertices"]
                cls.add_vertices(file, entity, close_file=close_file)

            elif attr == "octree_cells":
                del entity_handle["Octree Cells"]
                cls.add_octree_cells(file, entity, close_file=close_file)

            elif attr == "property_groups":
                del entity_handle["PropertyGroups"]
                cls.add_property_groups(file, entity, close_file=close_file)

            elif attr == "cell_delimiters":
                cls.add_cell_delimiters(file, entity, close_file=close_file)

            else:
                cls.add_attributes(file, entity, close_file=close_file)

    @staticmethod
    def uuid_value(value: str) -> uuid.UUID:
        return uuid.UUID(value)

    @staticmethod
    def uuid_str(value: uuid.UUID) -> str:
        return "{" + str(value) + "}"

    @staticmethod
    def bool_value(value: int8) -> bool:
        return bool(value)

    @staticmethod
    def fetch_h5_handle(file, entity) -> h5py.File:
        """
        fetch_h5_handle(file)

        Check if file reference to an existing hdf5

        Returns
        -------
        h5py: h5py.File
            Handle to an opened h5py file

        """
        if file is None:
            h5file = h5py.File(entity.workspace.h5file, "r+")

        else:
            if not isinstance(file, h5py.File):
                h5file = h5py.File(file, "r+")
            else:
                h5file = file

        return h5file
