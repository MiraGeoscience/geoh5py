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
        if self._attributes is None:
            self.attributes = getattr(self, "workspace").fetch_concatenated_values(
                self, "attributes"
            )

        return self._attributes

    @attributes.setter
    def attributes(self, attr: dict):
        if not isinstance(attr, dict):
            raise AttributeError("Input value for 'attributes' must be of type dict.")

        if "Attributes" in attr:
            attr = {uuid.UUID(elem["ID"]): elem for elem in attr["Attributes"]}

        self._attributes = attr

    @property
    def concatenated_object_ids(self):
        """Dictionary of concatenated objects and data concatenated_object_ids."""
        if getattr(self, "_concatenated_object_ids", None) is None:
            self._concatenated_object_ids = getattr(
                self, "workspace"
            ).fetch_concatenated_values(self, "concatenated_object_ids")
        return self._concatenated_object_ids

    @concatenated_object_ids.setter
    def concatenated_object_ids(self, object_ids: list[uuid.UUID]):
        if not isinstance(object_ids, list):
            raise AttributeError(
                "Input value for 'concatenated_object_ids' must be of type list."
            )

        self._concatenated_object_ids = object_ids

    @property
    def concatenator(self):
        return self

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

    def fetch_concatenated_objects(self):
        """
        Load all concatenated children.
        :param group: Concatenator group
        :param attributes: Entities stored as list of dictionaries.
        """
        attr_dict = {}
        for key in self.concatenated_object_ids:
            attrs = {
                attr: val
                for attr, val in self.attributes[key].items()
                if "Property" not in attr
            }
            attrs["parent"] = self
            attr_dict[key] = getattr(self, "workspace").create_from_concatenation(attrs)

        return attr_dict

    def fetch_values(self, entity, field: str):
        """
        Get values from a concatenated array.
        """
        if field not in self.data:
            data, indices = getattr(self, "workspace").fetch_concatenated_values(
                self, field
            )
            self.data[field] = data
            self.indices[field] = indices

        try:
            start, size = self.indices[field][entity.uid][:2]
        except KeyError:
            start, size = self.indices[field][entity.parent.uid][:2]

        return self.data[field][start : start + size]

    @property
    def property_group_ids(self):
        """Dictionary of concatenated objects and data property_group_ids."""
        if getattr(self, "_property_group_ids", None) is None:
            self._property_group_ids = getattr(
                self, "workspace"
            ).fetch_concatenated_values(self, "property_group_ids")
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

    @property
    def concatenator(self):
        """
        Parental Concatenator entity.
        """
        return self.parent.concatenator if self.parent is not None else None

    def find_or_create_property_group(self, **kwargs):
        """
        Overloaded :obj:`~geoh5py.objects.ObjectBase.find_or_create_property_group` method.
        """
        raise NotImplementedError(
            "Concatenated entity `find_or_create_property_group` method not yet implemented."
        )

    def fetch_values(self, entity, field: str):
        """
        Get values from the parent entity.
        """
        return self.concatenator.fetch_values(entity, field)

    def get_data(self, name: str) -> list:
        """
        Generic function to get data values from object.
        """
        entity_list = []
        uid = getattr(self, "uid")
        if f"Property:{name}" in self.parent.attributes[uid]:
            uid = self.parent.attributes[uid].get(f"Property:{name}")
            attributes = self.parent.attributes[uuid.UUID(uid)]
            attributes["parent"] = self
            getattr(self, "workspace").create_from_concatenation(attributes)

        for child in getattr(self, "children"):
            if hasattr(child, "name") and child.name == name:
                entity_list.append(child)

        return entity_list

    def get_data_list(self):
        """
        Get list of data names.
        """
        uid = getattr(self, "uid")
        data_list = [
            attr.replace("Property:", "")
            for attr in self.parent.attributes[uid]
            if "Property:" in attr
        ]

        return data_list

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
    #     values = getattr(self, "parent").fetch_values(getattr(self, "name"))
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
    #     surveys = self.fetch_values("Surveys")
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
    #     trace = self.fetch_values("Trace")
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
    #     trace_depth = self.fetch_values("TraceDepth")
    #
    #     return trace_depth

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        if (parent.concatenation is not Concatenated) and (
            parent.concatenation is not Concatenator
        ):
            raise AttributeError(
                "The 'parent' of a concatenated Entity must be of type 'Concatenator'."
            )
        self._parent = parent

        current_parent = self._parent

        if parent is not None:
            self._parent = parent
            self._parent.add_children([self])

            if current_parent is not None and current_parent != self._parent:
                current_parent.remove_children([self])
