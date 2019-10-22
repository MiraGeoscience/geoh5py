import uuid

import h5py
import numpy as np


class H5Writer:
    @staticmethod
    def bool_value(value: bool) -> np.uint8:
        return np.uint8(1 if value else 0)

    @staticmethod
    def uuid_value(value: uuid.UUID) -> str:
        return f"{{{value}}}"

    @classmethod
    def create_project(cls, file: str, workspace):
        str_type = h5py.special_dtype(vlen=str)
        h5file = h5py.File(file, "w")

        attr = workspace.get_workspace_attributes().__dict__
        project = h5file.create_group(workspace.base_name)
        project.attrs.create("Distance unit", attr["distance_unit"], dtype=str_type)
        project.attrs.create("Version", attr["version"])
        project.attrs.create("Contributors", attr["contributors"], dtype=str_type)
        project.attrs.create("GA Version", attr["ga_version"], dtype=str_type)

        # Create base entity structure for geoh5
        project.create_group("Data")
        project.create_group("Groups")
        project.create_group("Objects")
        types = project.create_group("Types")
        types.create_group("Data types")
        types.create_group("Group types")
        types.create_group("Object types")

        h5file.close()

    @classmethod
    def add_type(cls, file: str, tree: dict, uid: uuid.UUID):
        str_type = h5py.special_dtype(vlen=str)
        # Check if file reference to an opened hdf5
        if not isinstance(file, h5py.File):
            h5file = h5py.File(file, "r+")
        else:
            h5file = file

        base = list(h5file.keys())[0]

        entity_type = tree[uid]["entity_type"].replace("_", " ").capitalize() + "s"
        new_type = h5file[base]["Types"][entity_type].create_group("{" + str(uid) + "}")

        for key, value in tree[uid].items():
            if key == "entity_type":
                continue
            if key == "id":
                entry_key = key.upper()
            else:
                entry_key = key.capitalize()

            new_type.attrs.create(entry_key, value, dtype=str_type)

        if not isinstance(file, h5py.File):
            h5file.close()
            return None

        return new_type

    @classmethod
    def add_entity(cls, file: str, tree: dict, uid: uuid.UUID):
        str_type = h5py.special_dtype(vlen=str)
        # Check if file reference to an opened hdf5
        if not isinstance(file, h5py.File):
            h5file = h5py.File(file, "r+")
        else:
            h5file = file

        base = list(h5file.keys())[0]

        entity_type = tree[uid]["entity_type"].capitalize()

        if entity_type != "Data":
            entity_type += "s"

        new_entity = h5file[base][entity_type].create_group("{" + str(uid) + "}")

        for key, value in tree[uid].items():
            if key == "entity_type":
                continue
            if key == "id":
                entry_key = key.upper()
            else:
                entry_key = key.capitalize()

            new_entity.attrs.create(entry_key, value, dtype=str_type)

        new_type = H5Writer.add_type(h5file, tree, tree[uid]["type"])

        new_entity["Type"] = new_type

        # Check if file reference to an opened hdf5
        if not isinstance(file, h5py.File):
            h5file.close()
            return None

        return new_entity
