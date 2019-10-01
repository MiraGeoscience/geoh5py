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

        tree: dict = {
            "data": {},
            "objects": {},
            "groups": {},
            "types": {"data": {}, "group": {}, "object": {}},
        }

        data_types = project[base]["Types"]["Data types"]
        for uid in list(data_types.keys()):
            tree["types"]["data"][uid] = data_types[uid].attrs["Name"]

        group_types = project[base]["Types"]["Group types"]
        for uid in list(group_types.keys()):
            tree["types"]["group"][uid] = group_types[uid].attrs["Name"]

        object_types = project[base]["Types"]["Object types"]
        for uid in list(object_types.keys()):
            tree["types"]["object"][uid] = object_types[uid].attrs["Name"]

        data = project[base]["Data"]
        for uid in list(data.keys()):
            tree["data"][uid] = {}
            tree["data"][uid]["name"] = data[uid].attrs["Name"]
            tree["data"][uid]["type"] = data[uid]["Type"].attrs["ID"]
            tree["data"][uid]["parent"] = []
            tree["data"][uid]["children"] = []

        objects = project[base]["Objects"]
        for uid in list(objects.keys()):
            tree["objects"][uid] = {}
            tree["objects"][uid]["name"] = objects[uid].attrs["Name"]
            tree["objects"][uid]["type"] = objects[uid]["Type"].attrs["ID"]
            tree["objects"][uid]["parent"] = []
            children = list(objects[uid]["Data"].keys())

            # Assign as parent to data
            for child in children:
                tree["data"][objects[uid]["Data"][child].attrs["ID"]]["parent"] = uid

            tree["objects"][uid]["children"] = children

        groups = project[base]["Groups"]
        for uid in list(groups.keys()):
            tree["groups"][uid] = {}
            tree["groups"][uid]["name"] = groups[uid].attrs["Name"]
            tree["groups"][uid]["type"] = groups[uid]["Type"].attrs["ID"]
            tree["groups"][uid]["parent"] = []
            tree["groups"][uid]["children"] = []

            # Assign as parent to data
            for child in list(groups[uid]["Data"].keys()):
                tree["data"][groups[uid]["Data"][child].attrs["ID"]]["parent"] = uid

            children = list(groups[uid]["Data"].keys())

            # Assign as parent to objects
            for child in list(groups[uid]["Objects"].keys()):
                tree["objects"][groups[uid]["Objects"][child].attrs["ID"]][
                    "parent"
                ] = uid

            children += list(groups[uid]["Objects"].keys())

            # Check for parent group to groups
            for child in list(groups[uid]["Groups"].keys()):
                tree["groups"][groups[uid]["Groups"][child].attrs["ID"]]["parent"] = uid

            children += list(groups[uid]["Groups"].keys())

            tree["groups"][uid]["children"] = children

        project.close()

        return tree

    @staticmethod
    def bool_value(value: np.uint8) -> bool:
        return bool(value)

    @staticmethod
    def uuid_value(value: str) -> uuid.UUID:
        return uuid.UUID(value)
