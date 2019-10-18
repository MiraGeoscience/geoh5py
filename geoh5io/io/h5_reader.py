import uuid
from typing import Optional

import h5py
import numpy as np

from geoh5io.shared import Coord3D


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
    def get_vertices(cls, h5file: Optional[str], base: str, uid: uuid.UUID) -> Coord3D:
        project = h5py.File(h5file, "r+")

        x = project[base]["Objects"]["{" + str(uid) + "}"]["Vertices"]["x"]
        y = project[base]["Objects"]["{" + str(uid) + "}"]["Vertices"]["y"]
        z = project[base]["Objects"]["{" + str(uid) + "}"]["Vertices"]["z"]
        vertices = Coord3D((x, y, z))

        project.close()

        return vertices

    @classmethod
    def get_value(
        cls, h5file: Optional[str], base: str, uid: uuid.UUID
    ) -> Optional[float]:
        project = h5py.File(h5file, "r+")

        data = np.r_[project[base]["Data"]["{" + str(uid) + "}"]["Data"]]

        project.close()

        return data

    @classmethod
    def get_project_tree(cls, h5file: str, base: str) -> dict:

        project = h5py.File(h5file, "r+")

        tree: dict = {}

        # Load all entity types
        entity_type_classes = ["Data types", "Group types", "Object types"]
        for entity_type_class in entity_type_classes:
            for str_uid, entity_type in project[base]["Types"][
                entity_type_class
            ].items():
                uid = uuid.UUID(str_uid)
                tree[uid] = {}
                tree[uid]["entity_type"] = entity_type_class.replace(" ", "_").lower()[
                    :-1
                ]
                for key, value in entity_type.attrs.items():
                    tree[uid][key.replace(" ", "_").lower()] = value

        # Load all entities with relationships
        entity_classes = ["Data", "Objects", "Groups"]
        for entity_class in entity_classes:

            for str_uid, entity in project[base][entity_class].items():
                uid = uuid.UUID(str_uid)
                tree[uid] = {}
                tree[uid]["entity_type"] = entity_class.replace("s", "").lower()
                for key, value in entity.attrs.items():
                    tree[uid][key.replace(" ", "_").lower()] = value

                tree[uid]["type"] = uuid.UUID(entity["Type"].attrs["ID"])
                tree[uid]["parent"] = []
                tree[uid]["children"] = []

                if entity_class in ["Groups", "Objects"]:

                    # Assign as parent to data and data children
                    for key, value in entity["Data"].items():
                        tree[uuid.UUID(value.attrs["ID"])]["parent"] = uid
                        tree[uid]["children"] += [uuid.UUID(key)]

                if entity_class == "Groups":

                    # Assign as parent to data and data children
                    for key, value in entity["Objects"].items():
                        tree[uuid.UUID(value.attrs["ID"])]["parent"] = uid
                        tree[uid]["children"] += [uuid.UUID(key)]

        project.close()

        return tree

    @staticmethod
    def bool_value(value: np.uint8) -> bool:
        return bool(value)

    @staticmethod
    def uuid_value(value: str) -> uuid.UUID:
        return uuid.UUID(value)
