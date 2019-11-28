import uuid
from typing import Optional

import h5py
from numpy import c_, int8, ndarray, r_

from geoh5io.shared import Coord3D


class H5Reader:
    """
        H5 file Reader
    """

    @classmethod
    def get_project_attributes(cls, h5file: str, base: str) -> dict:
        """
        get_project_attributes(h5file, base)

        Get the project attributes

        Parameters
        ----------
        h5file: str
            Name of the project h5file

        base: str
            Name of the base project group ['GEOSCIENCE']

        Returns
        -------

        attributes: dict
            Dictionary of attributes
        """
        project = h5py.File(h5file, "r+")
        project_attrs = {}

        for key in list(project[base].attrs.keys()):

            attr = key.lower().replace(" ", "_")

            project_attrs[attr] = project[base].attrs[key]

        project.close()

        return project_attrs

    @classmethod
    def fetch_vertices(
        cls, h5file: Optional[str], base: str, uid: uuid.UUID
    ) -> Coord3D:
        """
        fetch_vertices(h5file, base, uid)

        Get the vertices of an object

        Parameters
        ----------
        h5file: str
            Name of the project h5file

        base: str
            Name of the base project group ['GEOSCIENCE']

        uid: uuid.UUID
            Unique identifier of the target object

        Returns
        -------
        vertices: geoh5io.Coord3D
            Coordinate object with vertex locations
        """
        project = h5py.File(h5file, "r+")

        x = project[base]["Objects"]["{" + str(uid) + "}"]["Vertices"]["x"]
        y = project[base]["Objects"]["{" + str(uid) + "}"]["Vertices"]["y"]
        z = project[base]["Objects"]["{" + str(uid) + "}"]["Vertices"]["z"]
        vertices = Coord3D(c_[x, y, z])

        project.close()

        return vertices

    @classmethod
    def fetch_cells(cls, h5file: Optional[str], base: str, uid: uuid.UUID) -> ndarray:
        """
        fetch_cells(h5file, base, uid)

        Get the cells of an object

        Parameters
        ----------
        h5file: str
            Name of the project h5file

        base: str
            Name of the base project group ['GEOSCIENCE']

        uid: uuid.UUID
            Unique identifier of the target object

        Returns
        -------
        cells: geoh5io.Cell
            Cell object with vertex indices defining the cell
        """
        project = h5py.File(h5file, "r+")

        indices = project[base]["Objects"]["{" + str(uid) + "}"]["Cells"][:]

        project.close()

        return indices

    @classmethod
    def fetch_values(
        cls, h5file: Optional[str], base: str, uid: uuid.UUID
    ) -> Optional[float]:
        """
        fetch_values(h5file, base, uid)

        Get the values of an entity

        Parameters
        ----------
        h5file: str
            Name of the project h5file

        base: str
            Name of the base project group ['GEOSCIENCE']

        uid: uuid.UUID
            Unique identifier of the target entity

        Returns
        -------
        values: numpy.array
            Array of values
        """

        project = h5py.File(h5file, "r+")

        values = r_[project[base]["Data"]["{" + str(uid) + "}"]["Data"]]

        project.close()

        return values

    @classmethod
    def get_project_tree(cls, h5file: str, base: str) -> dict:
        """
        get_project_tree(h5file, base)

        Get the values of an entity

        Parameters
        ----------
        h5file: str
            Name of the project h5file

        base: str
            Name of the base project group ['GEOSCIENCE']

        Returns
        -------
        tree: dict
            Dictionary of group, objects, data and types found in the geoh5 file.
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

                    # Assign as parent to data and data children
                    for key, value in entity["Groups"].items():
                        tree[uuid.UUID(value.attrs["ID"])]["parent"] = uid
                        tree[uid]["children"] += [uuid.UUID(key)]

        project.close()

        return tree

    @staticmethod
    def bool_value(value: int8) -> bool:
        return bool(value)

    @staticmethod
    def uuid_value(value: str) -> uuid.UUID:
        return uuid.UUID(value)
