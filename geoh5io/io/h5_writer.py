import uuid

import h5py
import numpy as np


class H5Writer:

    str_type = h5py.special_dtype(vlen=str)

    @staticmethod
    def bool_value(value: bool) -> np.uint8:
        return np.uint8(1 if value else 0)

    @staticmethod
    def uuid_value(value: uuid.UUID) -> str:
        return f"{{{value}}}"

    @classmethod
    def create_geoh5(cls, file: str, workspace, close_file=True):

        # Check if file reference to an opened hdf5
        h5file = h5py.File(file, "w")

        attr = workspace.get_workspace_attributes().__dict__
        project = h5file.create_group(workspace.base_name)
        project.attrs.create("Distance unit", attr["distance_unit"], dtype=cls.str_type)
        project.attrs.create("Version", attr["version"])
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

        H5Writer.add_entity(
            file, workspace.tree, workspace.get_entity("Workspace")[0].uid
        )

        if close_file:
            h5file.close()
            return None
        return h5file

    @classmethod
    def add_type(cls, file: str, tree: dict, uid: uuid.UUID, close_file=True):

        # Check if file reference to an opened hdf5
        if not isinstance(file, h5py.File):
            h5file = h5py.File(file, "r+")
        else:
            h5file = file

        base = list(h5file.keys())[0]

        entity_type = tree[uid]["entity_type"].replace("_", " ").capitalize() + "s"
        new_type = h5file[base]["Types"][entity_type].create_group(cls.uuid_value(uid))

        for key, value in tree[uid].items():
            if key == "entity_type":
                continue
            if key == "id":
                entry_key = key.upper()
            else:
                entry_key = key.capitalize()

            new_type.attrs.create(entry_key, value, dtype=cls.str_type)

        if close_file:
            h5file.close()
            return None
        return new_type

    @classmethod
    def add_entity(
        cls, file: str, tree: dict, uid: uuid.UUID, values=None, close_file=True
    ):
        cls.str_type = h5py.special_dtype(vlen=str)
        # Check if file reference to an opened hdf5
        if not isinstance(file, h5py.File):
            h5file = h5py.File(file, "r+")
        else:
            h5file = file

        base = list(h5file.keys())[0]

        entity_type = tree[uid]["entity_type"].capitalize()

        if entity_type != "Data":
            entity_type += "s"

        new_entity = h5file[base][entity_type].create_group(cls.uuid_value(uid))

        for key, value in tree[uid].items():
            if key in ["type", "entity_type", "parent", "children"]:
                continue
            if key == "id":
                entry_key = key.upper()
            else:
                entry_key = key.capitalize()

            if isinstance(value, str):
                new_entity.attrs.create(entry_key, value, dtype=cls.str_type)
            else:
                new_entity.attrs.create(entry_key, value, dtype="int8")

        # Add the type and return a pointer
        new_type = H5Writer.add_type(h5file, tree, tree[uid]["type"], close_file=False)

        new_entity["Type"] = new_type

        if values is not None:
            new_entity["Data"] = values

        # Check if file reference to an opened hdf5
        if close_file:
            h5file.close()
            return None
        return new_entity
