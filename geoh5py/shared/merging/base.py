#  Copyright (c) 2024 Mira Geoscience Ltd.
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

from abc import ABC, abstractmethod
from warnings import warn

import numpy as np

from ...data import NumericData
from ...objects import ObjectBase
from ...workspace import Workspace


class BaseMerger(ABC):
    _type: type = ObjectBase

    @classmethod
    def merge_data(
        cls,
        out_entity,
        input_entities: list,
    ):
        """
        Merge the data respecting the entity type, the values, and the association.
        :param out_entity: the output entity to add the data to.
        :param input_entities: the list of objects to merge the data from.
        :return: a dictionary of data to add to the merged object.
        """
        data_dict: dict[tuple, NumericData] = {}
        data_count = {"VERTEX": 0, "CELL": 0}

        for input_entity in input_entities:
            for ind, data in enumerate(input_entity.children):
                if (
                    not isinstance(data, NumericData)
                    or data.association is None
                    or data.n_values is None
                    or data.values is None
                ):
                    continue

                association = data.association.name
                label = (data.name, data.entity_type.name, association)
                start, end = (
                    data_count[association],
                    data_count[association] + data.n_values,
                )

                # Increment label in case of duplicate match
                if label in data_dict:
                    values = data_dict[label].values
                    if isinstance(values, np.ndarray) and (
                        ~np.all(
                            np.logical_or(
                                values[start:end] == data.nan_value,
                                np.isnan(values[start:end]),
                            )
                        )
                    ):
                        label = (
                            data.name + f"({ind})",
                            data.entity_type.name,
                            association,
                        )
                        warn(
                            f"Multiple data '{data.name}' with entity_type "
                            f"name '{data.entity_type.name}' "
                            f"were found on object '{input_entity.name}'. "
                            f"The merging operation is ambiguous. "
                            f"Please validate or rename the data."
                        )

                if label not in data_dict:
                    shape = (
                        out_entity.n_vertices
                        if association == "VERTEX"
                        else out_entity.n_cells
                    )

                    data_dict[label] = out_entity.add_data(
                        {
                            data.name: {
                                "values": np.ones(shape) * data.nan_value,
                                "association": association,
                                "entity_type": data.entity_type,
                            }
                        }
                    )

                values = data_dict[label].values

                if isinstance(values, np.ndarray):
                    values[start:end] = data.values
                    data_dict[label].values = values

            data_count["VERTEX"] += getattr(input_entity, "n_vertices", 0) or 0
            data_count["CELL"] += getattr(input_entity, "n_cells", 0) or 0

    @classmethod
    def merge_objects(
        cls,
        workspace: Workspace,
        input_entities: list[ObjectBase],
        add_data: bool = True,
        **kwargs,
    ):
        """
        Merge a list of :obj:geoh5py.objects.ObjectBase of the same type.

        :param workspace: the workspace to use.
        :param input_entities: the list of objects to merge.
        :param add_data: if True, the data will be merged.
        :param kwargs: the kwargs to create the new object.
        :return: an object of type input_entities[0] containing the merged vertices.
        """
        cls.validate_objects(input_entities)
        kwargs.pop("workspace", None)

        output = cls.create_object(workspace, input_entities, **kwargs)

        # add the data
        if add_data:
            cls.merge_data(output, input_entities)

        return output

    @classmethod
    @abstractmethod
    def create_object(cls, workspace: Workspace, input_entities, **kwargs):
        """
        Create an object with the structure of all the input entities.
        :param workspace: The workspace to use to create the object.
        :param input_entities: all the input entities to merge.
        :param kwargs: the kwargs to create the new object.
        :return: the new object.
        """
        raise NotImplementedError("BaseMerger cannot be use, use a subclass.")

    @classmethod
    def validate_objects(cls, input_entities: list[ObjectBase]):
        """
        Validate the input entities types and raises error if incompatible.

        :param input_entities: a list of :obj:geoh5py.objects.ObjectBase objects.
        """
        # assert input entities is a list of len superior to 1
        if not isinstance(input_entities, list):
            raise TypeError("The input entities must be a list of geoh5py objects.")

        # assert input entities is a list of len superior to 1
        if len(input_entities) < 2:
            raise ValueError("Need more than one object to merge.")

        # assert input entities are of the same type
        if not all(
            type(input_entity) is type(input_entities[0])
            for input_entity in input_entities
        ):
            raise TypeError("All objects must be of the same type.")

        # assert input entities are of the same type
        cls.validate_type(input_entities[0])

        # verify if the all input entities have vertices
        for input_entity in input_entities:
            cls.validate_structure(input_entity)

    @classmethod
    @abstractmethod
    def validate_structure(cls, input_entity):
        """
        Validate the input entity structure and raises error if incompatible.
        :param input_entity: the input entity to validate.
        """
        raise NotImplementedError("BaseMerger cannot be use, use a subclass.")

    @classmethod
    def validate_type(cls, input_entity):
        if type(input_entity) is not cls._type:  # pylint: disable=unidiomatic-typecheck
            raise TypeError(
                f"The input entities must be a list of {cls._type.__name__}."
            )
