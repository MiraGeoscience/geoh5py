#  Copyright (c) 2023 Mira Geoscience Ltd.
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
from typing import cast

import numpy as np

from ...data import Data
from ...objects import ObjectBase
from ...workspace import Workspace


class BaseMerger(ABC):
    _type: type = ObjectBase

    def __init__(self):
        """
        Merging class for :obj:geoh5py.objects.ObjectBase of the same types.
        """

    @classmethod
    def merge_data(cls, input_entities: list[ObjectBase]) -> dict[str, dict]:
        """
        Merge the data respecting the entity type, the values, and the association.
        :param input_entities:
        :return: a dictionary of data to add to the merged object.
        """

        # extract the data from the input entities
        id_, previous, nb_keys = 0, 0, 1
        all_data, unique_entity_types = [], []
        for input_entity in input_entities:
            entity_unique_entity_types = []

            for data in input_entity.children:
                if not isinstance(data, Data):
                    continue

                all_data.append(
                    [
                        data.values,
                        (data.entity_type, data.name, id_),
                        data.association,
                        previous,
                        previous + cast(int, input_entity.n_vertices),
                    ]
                )
                entity_unique_entity_types.append((data.entity_type, data.name))
                unique_entity_types.append((data.entity_type, data.name, id_))
                id_ += 1

            # define how to define the entity_type
            if nb_keys != 3 and len(
                list({unitype[:nb_keys] for unitype in entity_unique_entity_types})
            ) != len(entity_unique_entity_types):
                nb_keys = 2
                if len(list(set(entity_unique_entity_types))) != len(
                    entity_unique_entity_types
                ):
                    nb_keys = 3

            previous += cast(int, input_entity.n_vertices)

        # create an intermediate array
        all_data = np.array(all_data, dtype=object)

        # prepare the attributes
        unique_entity_types = list(
            dict.fromkeys(
                [entity_type[:nb_keys] for entity_type in unique_entity_types]  # type: ignore
            )
        )
        unique_associations: dict = {
            entity_type: [] for entity_type in unique_entity_types
        }
        data_array = np.empty((len(unique_entity_types), previous))
        data_array[:] = np.nan

        # populate the intermediate array
        for data in all_data:
            data_array[
                unique_entity_types.index(data[1][:nb_keys]), data[-2] : data[-1]
            ] = data[0]
            unique_associations[data[1][:nb_keys]].append(data[2])

        # create the output data
        add_data_dict = {}
        for id_, data in enumerate(data_array):
            # define the association
            association = unique_associations[unique_entity_types[id_]]

            if not len(set(association)) == 1:
                raise ValueError(
                    f"Cannot merge data with different associations: {set(association)}"
                )

            # define the name
            if nb_keys == 3:
                name = f"merged_{unique_entity_types[id_][0].name}_{id_}"
            elif nb_keys == 2:
                name = f"merged_{unique_entity_types[id_][0].name}_{unique_entity_types[id_][1]}"
            else:
                name = f"merged_{unique_entity_types[id_][0].name}"

            add_data_dict[name] = {
                "association": association[0],
                "values": data,
                "entity_type": unique_entity_types[id_][0],
            }

        return add_data_dict

    @classmethod
    def merge_objects(
        cls, input_entities: list[ObjectBase], add_data: bool = True, **kwargs
    ):
        """
        Merge a list of :obj:geoh5py.objects.ObjectBase of the same type.
        :param input_entities: the list of objects to merge.
        :param add_data: if True, the data will be merged.
        :param kwargs: the kwargs to create the new object.
        :return: an object of type input_entities[0] containing the merged vertices.
        """
        cls.validate_objects(input_entities)

        kwargs = cls.validate_kwargs(input_entities[0].workspace, **kwargs)

        # create the vertices
        vertices = np.vstack([input_entity.vertices for input_entity in input_entities])

        # create an object of type input_entities[0]
        output_type = type(input_entities[0])
        output = output_type.create(vertices=vertices, **kwargs)

        # add the data
        if add_data:
            output.add_data(cls.merge_data(input_entities))

        return output

    @classmethod
    def validate_kwargs(cls, workspace: Workspace, **kwargs) -> dict:
        """
        Validate the kwargs to create a new object.
        :param workspace: the workspace to eventually use.
        :param kwargs: the kwargs to create the new object.
        :return: the validated kwargs.
        """

        if "workspace" not in kwargs:
            kwargs["workspace"] = workspace

        return kwargs

    @classmethod
    @abstractmethod
    def validate_type(cls, input_entity: ObjectBase):
        raise NotImplementedError("BaseMerger cannot be used.")

    @classmethod
    def validate_objects(cls, input_entities: list[ObjectBase]):
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
        if not all(
            isinstance(input_entity.vertices, np.ndarray)
            for input_entity in input_entities
        ):
            raise AttributeError("All entities must have vertices.")
