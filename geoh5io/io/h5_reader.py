import uuid

import h5py
import numpy as np


class H5Reader:
    """
        H5 file Reader
    """

    @classmethod
    def get_project_attributes(cls, h5file: str, base: str) -> dict:

        project = h5py.File(h5file, "r+")
        project_attrs = {}

        for key in list(project[base].attrs.keys()):

            attr = key.lower().replace(" ", "_")

            project_attrs[attr] = project[base].attrs[key]

        project.close()

        return project_attrs

    @classmethod
    def get_project_tree(cls, h5file: str, base: str) -> dict:

        project = h5py.File(h5file, "r+")

        tree: dict = {"data": {}, "groups": {}, "objects": {}, "types": {}}

        data = project[base]["Data"]
        for key in list(data.keys()):
            uid = uuid.UUID(key)
            tree["data"][uid] = {}
            tree["data"][uid]["name"] = data[key].attrs["Name"]
            tree["data"][uid]["type"] = uuid.UUID(data[key]["Type"].attrs["ID"])

        project.close()

        return tree

    @staticmethod
    def bool_value(value: np.uint8) -> bool:
        return bool(value)

    @staticmethod
    def uuid_value(value: str) -> uuid.UUID:
        return uuid.UUID(value)
