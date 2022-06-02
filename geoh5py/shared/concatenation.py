#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import uuid

# from numpy import ndarray, vstack


class Concatenator:
    """
    Class modifier for concatenation of objects and data.
    """

    _attribute_map = {
        "Attributes": "attributes",
        "Property Groups IDs": "property_group_ids",
        "Concatenated object IDs": "concatenated_object_ids",
        "Concatenated Data": "concatenated_data",
    }
    _concatenated_data = None
    _concatenated_object_ids = None
    _property_group_ids = None
    _property_groups = None
    _attributes = None
    _data: dict = {}
    _indices: dict = {}

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    @property
    def attributes(self):
        """Dictionary of concatenated objects and data attributes."""
        return self._attributes

    @attributes.setter
    def attributes(self, attr: dict):
        if not isinstance(attr, dict):
            raise AttributeError("Input value for 'attributes' must be of type dict.")

        self._attributes = attr

    @property
    def concatenated_object_ids(self):
        """Dictionary of concatenated objects and data concatenated_object_ids."""
        if getattr(self, "_concatenated_object_ids", None) is None:
            self._concatenated_object_ids = getattr(
                self, "workspace"
            ).fetch_concatenated_data(self, "Concatenated object IDs")
        return self._concatenated_object_ids

    @concatenated_object_ids.setter
    def concatenated_object_ids(self, object_ids: list[uuid.UUID]):
        if not isinstance(object_ids, list):
            raise AttributeError(
                "Input value for 'concatenated_object_ids' must be of type list."
            )

        self._concatenated_object_ids = object_ids

    @property
    def data(self) -> dict:
        """
        Concatenated data values stored as a dictionary.
        """
        return self._data

    @property
    def indices(self) -> dict:
        """
        Concatenated indices stored as a dictionary.
        """
        return self._indices

    def get_concatenated_data(self, entity, field: str):
        """
        Get values from a concatenated array.
        """
        if field not in self.data:
            data, indices = getattr(self, "workspace").fetch_concatenated_data(
                self, field
            )
            self.data[field] = data
            self.indices[field] = indices

        start, size = self.indices[field][entity.uid][:2]
        return self.data[field][start : start + size]

    @property
    def property_group_ids(self):
        """Dictionary of concatenated objects and data property_group_ids."""
        if getattr(self, "_property_group_ids", None) is None:
            self._property_group_ids = getattr(
                self, "workspace"
            ).fetch_concatenated_data(self, "Property Group IDs")
        return self._property_group_ids

    @property_group_ids.setter
    def property_group_ids(self, object_ids: list[uuid.UUID]):
        if not isinstance(object_ids, list):
            raise AttributeError(
                "Input value for 'property_group_ids' must be of type list."
            )

        self._property_group_ids = object_ids

    @property
    def property_groups(self):
        """
        Property groups for the concatenated data.
        """
        if getattr(self, "_property_groups", None) is None:
            self._property_groups = getattr(self, "workspace").fetch_property_groups(
                self
            )

        return self._property_groups

    def update_attributes(self, entity, field):
        """Update the attributes of a concatenated entity."""

        if field == "attributes":
            for key in self.attributes[entity.uid]:
                self.attributes[entity.uid][key] = getattr(entity, key)

    # def add_children(self, children):
    #     """
    #     :param children: Add a list of entities as
    #         :obj:`~geoh5py.shared.entity.Entity.children`
    #     """
    #     for child in children:
    #         if child not in self._children:
    #             self._children.append(child)


class Concatenated:
    """
    Class modifier for concatenated objects and data.
    """

    def __init__(self, **kwargs):
        self._parent: Concatenated | Concatenator | None = None

        super().__init__(**kwargs)

    def add_data(self, data: dict, property_group: str = None):
        """
        Overloaded :obj:`~geoh5py.objects.ObjectBase.add_data` method.
        """
        raise NotImplementedError(
            "Concatenated entity `add_data` method not yet implemented."
        )

    def add_data_to_group(self, data: dict, property_group: str = None):
        """
        Overloaded :obj:`~geoh5py.objects.ObjectBase.add_data_to_group` method.
        """
        raise NotImplementedError(
            "Concatenated entity `add_data_to_group` method not yet implemented."
        )

    def get_data_list(self):
        """
        Get list of data names.
        """
        data_list = []
        for attr in self.__dict__:
            if "Property:" in attr:
                data_list.append(attr.replace("Property:", ""))

        return data_list

    def find_or_create_property_group(self, **kwargs):
        """
        Overloaded :obj:`~geoh5py.objects.ObjectBase.find_or_create_property_group` method.
        """
        raise NotImplementedError(
            "Concatenated entity `find_or_create_property_group` method not yet implemented."
        )

    def get_data(self, name: str) -> list:
        """
        Generic function to get data values from object.
        """
        entity_list = []

        if f"Property:{name}" in self.__dict__:
            if isinstance(getattr(self, f"Property:{name}"), str):
                uid = getattr(self, f"Property:{name}")
                attributes = getattr(self, "parent").attributes[uuid.UUID(uid)]
                attributes["parent"] = self
                getattr(self, "workspace").create_concatenated_entity(attributes)

        for child in getattr(self, "children"):
            if hasattr(child, "name") and child.name == name:
                entity_list.append(child)

        return entity_list

    @staticmethod
    def get_concatenated_data(entity, field: str):
        """
        Get values from the parent entity.
        """
        return getattr(entity, "parent").get_concatenated_data(entity, field)

    @staticmethod
    def update_attributes(entity, field: str):
        """
        Update the attributes on the concatenated entity.
        """
        return getattr(entity, "parent").update_attributes(entity, field)

    # @property
    # def values(self) -> ndarray:
    #     """
    #     :return: values: An array of float values
    #     """
    #     if not hasattr(self, "_values"):
    #         raise AttributeError(f"Concatenated entity {self} does not have values.")
    #
    #     values = getattr(self, "parent").get_concatenated_data(getattr(self, "name"))
    #
    #     if values is not None:
    #         values = getattr(self, "check_vector_length")(values)
    #
    #     return values
    #
    # @property
    # def surveys(self):
    #     """
    #     :obj:`numpy.array` of :obj:`float`, shape (3, ): Coordinates of the surveys
    #     """
    #     surveys = self.get_concatenated_data("Surveys")
    #
    #     if surveys is not None:
    #         surveys = vstack([surveys["Depth"], surveys["Azimuth"], surveys["Dip"]]).T
    #         surveys = vstack([surveys[0, :], surveys])
    #         surveys[0, 0] = 0.0
    #
    #         return surveys.astype(float)
    #
    #     return None
    #
    # @property
    # def trace(self) -> ndarray | None:
    #     """
    #     :obj:`numpy.array`: Drillhole trace defining the path in 3D
    #     """
    #     trace = self.get_concatenated_data("Trace")
    #
    #     if trace is not None:
    #         return trace.view("<f8").reshape((-1, 3))
    #
    #     return None
    #
    # @property
    # def trace_depth(self) -> ndarray | None:
    #     """
    #     :obj:`numpy.array`: Drillhole trace depth from top to bottom
    #     """
    #     trace_depth = self.get_concatenated_data("TraceDepth")
    #
    #     return trace_depth

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, entity):
        if (entity.concatenation is not Concatenated) and (
            entity.concatenation is not Concatenator
        ):
            raise AttributeError(
                "The 'parent' of a concatenated entity must be of type 'Concatenator'."
            )
        self._parent = entity
