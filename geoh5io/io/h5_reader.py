import uuid
from typing import Any, Dict, Optional, Tuple

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
        project = h5py.File(h5file, "r")
        project_attrs = {}

        for key in project[base].attrs.keys():

            attr = key.lower().replace(" ", "_")

            project_attrs[attr] = project[base].attrs[key]

        project.close()

        return project_attrs

    @classmethod
    def fetch_project_attributes(cls, h5file: str) -> Dict[Any, Any]:
        """
        fetch_project_attributes(h5file)

        Get attributes og object from geoh5

        Parameters
        ----------
        h5file: str
            Name of the project h5file

        Returns
        -------
        attributes: dict
            Dictionary of attributes from geoh5
        """
        project = h5py.File(h5file, "r")
        name = list(project.keys())[0]
        attributes = {}

        for key, value in project[name].attrs.items():
            attributes[key] = value

        project.close()

        return attributes

    @classmethod
    def fetch_attributes(
        cls, h5file: str, name: str, uid: uuid.UUID, entity_type: str
    ) -> Tuple[dict, dict, dict]:
        """
        fetch_attributes(h5file, name, uid, entity_type)

        Get attributes og object from geoh5

        Parameters
        ----------
        h5file: str
            Name of the project h5file

        name: str
            Name of the project group ['GEOSCIENCE']

        uid: uuid.UUID
            Unique identifier

        entity_type: str
            Type of entity from "group", "data", "object", "group_type", "data_type", "object_type"

        Returns
        -------
        attributes: dict
            Dictionary of attributes from geoh5
        """
        project = h5py.File(h5file, "r")
        attributes: Dict = {"entity": {}}
        type_attributes: Dict = {"entity_type": {}}
        property_groups: Dict = {}
        if "type" in entity_type:
            entity_type = entity_type.replace("_", " ").capitalize() + "s"
            entity = project[name]["Types"][entity_type][cls.uuid_str(uid)]
        elif entity_type == "Root":
            entity = project[name][entity_type]
        else:
            entity_type = entity_type.capitalize()
            if entity_type in ["Group", "Object"]:
                entity_type += "s"
            entity = project[name][entity_type][cls.uuid_str(uid)]

        for key, value in entity.attrs.items():
            attributes["entity"][key] = value

        for key, value in entity["Type"].attrs.items():
            type_attributes["entity_type"][key] = value

        # Check if the entity has property_group
        if "PropertyGroups" in entity.keys():
            for pg_id in entity["PropertyGroups"].keys():
                property_groups[pg_id] = {"uid": pg_id}
                for key, value in entity["PropertyGroups"][pg_id].attrs.items():
                    property_groups[pg_id][key] = value

        project.close()

        attributes["entity"]["existing_h5_entity"] = True
        return attributes, type_attributes, property_groups

    @classmethod
    def fetch_children(
        cls, h5file: str, base: str, uid: uuid.UUID, entity_type: str
    ) -> dict:
        """
        fetch_children(h5file, base, uid, entity_type)

        Get children of object from geoh5

        Parameters
        ----------
        h5file: str
            Name of the project h5file

        base: str
            Name of the base project group ['GEOSCIENCE']

        uid: uuid.UUID
            Unique identifier

        entity_type: str
            Type of entity from "group", "data", "object", "group_type", "data_type", "object_type"

        Returns
        -------
        children: list of dict
            List of dictionaries for the children uid and type
        """
        project = h5py.File(h5file, "r")

        children = {}
        entity_type = entity_type.capitalize()
        if entity_type in ["Group", "Object"]:
            entity_type += "s"
        entity = project[base][entity_type][cls.uuid_str(uid)]

        for child_type, child_list in entity.items():
            if child_type in ["Type", "PropertyGroups"]:
                continue

            if isinstance(child_list, h5py.Group):
                for uid_str in child_list.keys():
                    children[cls.uuid_value(uid_str)] = child_type.replace(
                        "s", ""
                    ).lower()

        project.close()

        return children

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
        project = h5py.File(h5file, "r")

        x = project[base]["Objects"][cls.uuid_str(uid)]["Vertices"]["x"]
        y = project[base]["Objects"][cls.uuid_str(uid)]["Vertices"]["y"]
        z = project[base]["Objects"][cls.uuid_str(uid)]["Vertices"]["z"]
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
        project = h5py.File(h5file, "r")

        indices = project[base]["Objects"][cls.uuid_str(uid)]["Cells"][:]

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
        project = h5py.File(h5file, "r")

        if "Data" in list(project[base]["Data"][cls.uuid_str(uid)].keys()):
            values = r_[project[base]["Data"][cls.uuid_str(uid)]["Data"]]
        else:
            values = None

        project.close()

        return values

    @classmethod
    def fetch_delimiters(
        cls, h5file: Optional[str], base: str, uid: uuid.UUID
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """
        fetch_delimiters(h5file, base, uid)

        Get the delimiters of an entity

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
        u_delimiters: numpy.array
            Array of u_delimiters

        v_delimiters: numpy.array
            Array of v_delimiters

        z_delimiters: numpy.array
            Array of z_delimiters
        """
        project = h5py.File(h5file, "r")

        u_delimiters = r_[
            project[base]["Objects"][cls.uuid_str(uid)]["U cell delimiters"]
        ]
        v_delimiters = r_[
            project[base]["Objects"][cls.uuid_str(uid)]["V cell delimiters"]
        ]
        z_delimiters = r_[
            project[base]["Objects"][cls.uuid_str(uid)]["Z cell delimiters"]
        ]

        project.close()

        return u_delimiters, v_delimiters, z_delimiters

    @classmethod
    def fetch_octree_cells(
        cls, h5file: Optional[str], base: str, uid: uuid.UUID
    ) -> ndarray:
        """
        fetch_octree_cells(h5file, base, uid)

        Get the octree_cells of an octree mesh

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
        octree_cells: numpy.ndarray(int)
            Array of octree_cells

        """
        project = h5py.File(h5file, "r")

        octree_cells = r_[project[base]["Objects"][cls.uuid_str(uid)]["Octree Cells"]]

        project.close()

        return octree_cells

    @classmethod
    def fetch_property_groups(
        cls, h5file: Optional[str], base: str, uid: uuid.UUID
    ) -> Dict[str, Dict[str, str]]:
        """
        fetch_property_groups(h5file, base, uid)

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
        property_group_attributes: dict
            Dictionary of property_groups attributes

        """
        project = h5py.File(h5file, "r")

        pg_handle = project[base]["Objects"][cls.uuid_str(uid)]["PropertyGroups"]

        property_groups: Dict[str, Dict[str, str]] = {}
        for pg_uid in pg_handle.keys():

            property_groups[pg_uid] = {}
            for attr, value in pg_handle[pg_uid].attrs.items():
                property_groups[pg_uid][attr] = value

        return property_groups

    # @classmethod
    # def get_project_tree(cls, h5file: str, base: str) -> dict:
    #     """
    #     get_project_tree(h5file, base)
    #
    #     Get the values of an entity
    #
    #     Parameters
    #     ----------
    #     h5file: str
    #         Name of the project h5file
    #
    #     base: str
    #         Name of the base project group ['GEOSCIENCE']
    #
    #     Returns
    #     -------
    #     tree: dict
    #         Dictionary of group, objects, data and types found in the geoh5 file.
    #         Used for light reference to attributes, parent and children.
    #         {uuid:
    #             {'name': str},
    #             {'entity_type': str},
    #             {'parent': uuid},
    #             {'children': [uuid1, uuid2,....],
    #          ...
    #          }
    #     """
    #     project = h5py.File(h5file, "r")
    #
    #     tree: dict = {}
    #
    #     # Load all entity types
    #     entity_type_classes = ["Data types", "Group types", "Object types"]
    #     for entity_type_class in entity_type_classes:
    #         for str_uid, entity_type in project[base]["Types"][
    #             entity_type_class
    #         ].items():
    #             uid = uuid.UUID(str_uid)
    #             tree[uid] = {}
    #             tree[uid]["entity_type"] = entity_type_class.replace(" ", "_").lower()[
    #                 :-1
    #             ]
    #             # for key, value in entity_type.attrs.items():
    #             tree[uid]["name"] = entity_type.attrs["Name"]
    #
    #     # Load all entities with relationships
    #     entity_classes = ["Data", "Objects", "Groups"]
    #     for entity_class in entity_classes:
    #
    #         for str_uid, entity in project[base][entity_class].items():
    #             uid = uuid.UUID(str_uid)
    #
    #             if uid not in tree.keys():
    #                 tree[uid] = {}
    #                 tree[uid]["parent"] = []
    #
    #             tree[uid]["entity_type"] = entity_class.replace("s", "").lower()
    #             # for key, value in entity.attrs.items():
    #             tree[uid]["name"] = entity.attrs["Name"]
    #
    #             tree[uid]["type"] = uuid.UUID(entity["Type"].attrs["ID"])
    #             tree[uid]["children"] = []
    #
    #             if entity_class in ["Groups", "Objects"]:
    #
    #                 # Assign as parent to data and data children
    #                 for key, value in entity["Data"].items():
    #                     if uuid.UUID(value.attrs["ID"]) not in tree.keys():
    #                         tree[uuid.UUID(value.attrs["ID"])] = {}
    #
    #                     tree[uuid.UUID(value.attrs["ID"])]["parent"] = uid
    #                     tree[uid]["children"] += [uuid.UUID(key)]
    #
    #             if entity_class == "Groups":
    #
    #                 # Assign as parent to data and data children
    #                 for key, value in entity["Objects"].items():
    #                     if uuid.UUID(value.attrs["ID"]) not in tree.keys():
    #                         tree[uuid.UUID(value.attrs["ID"])] = {}
    #
    #                     tree[uuid.UUID(value.attrs["ID"])]["parent"] = uid
    #                     tree[uid]["children"] += [uuid.UUID(key)]
    #
    #                 # Assign as parent to data and data children
    #                 for key, value in entity["Groups"].items():
    #                     if uuid.UUID(value.attrs["ID"]) not in tree.keys():
    #                         tree[uuid.UUID(value.attrs["ID"])] = {}
    #
    #                     tree[uuid.UUID(value.attrs["ID"])]["parent"] = uid
    #                     tree[uid]["children"] += [uuid.UUID(key)]
    #
    #     project.close()
    #
    #     return tree

    @staticmethod
    def bool_value(value: int8) -> bool:
        return bool(value)

    @staticmethod
    def uuid_value(value: str) -> uuid.UUID:
        return uuid.UUID(value)

    @staticmethod
    def uuid_str(value: uuid.UUID) -> str:
        return "{" + str(value) + "}"
