# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import numpy as np

from geoh5py.data import Data
from geoh5py.groups import Group, PropertyGroup
from geoh5py.objects import ObjectBase
from geoh5py.shared.utils import (
    dict_mapper,
    entity2uuid,
    str2uuid,
    str_json_to_dict,
    stringify,
    uuid2entity,
)


class UIJsonGroup(Group):
    """
    Group for storing ui.json files.

    In Geoscience ANALYST, it can be used to store ui.jsons.
    """

    _TYPE_UID = UUID("{BB50AC61-A657-4926-9C82-067658E246A0}")
    _default_name = "UIJson"
    _omit_list = ["_uijson_objects", "_uijson_children", "_uijson_groups", "_options"]

    def __init__(
        self,
        options: dict | np.ndarray | bytes | None = None,
        **kwargs,
    ):
        self._options: dict

        super().__init__(**kwargs)

        self.options = self.format_input_options(options)
        self._uijson_objects: list[ObjectBase] | None = None
        self._uijson_children: list[Data | PropertyGroup] | None = None
        self._uijson_groups: list[Group] | None = None

    @staticmethod
    def _copy_cherry_pick_object(
        obj: ObjectBase,
        parent,
        clear_cache: bool,
        cherry_pick: list[UUID],
    ) -> tuple[ObjectBase, dict[UUID, Data | PropertyGroup]]:
        """
        Copy an object and return a map of the copied children.

        :param obj: The object to copy.
        :param parent: The parent of the copied object.
        :param clear_cache: Indicate whether to clear the cache.
        :param cherry_pick: The list of uids to copy.

        :return: The copied object and a map of the copied children.
        """
        new_obj = obj.copy(
            parent=parent,
            copy_children=False,
            clear_cache=clear_cache,
            cherry_pick_uids=cherry_pick,
        )

        # WARNING: We are relying on name uniqueness here
        ui_children_map = {}
        if obj.property_groups:
            for prop in obj.property_groups:
                if prop.uid in cherry_pick:
                    ui_children_map[prop.uid] = new_obj.get_property_group(prop.name)[0]

        for child in obj.children:
            if isinstance(child, Data) and child.uid in cherry_pick:
                ui_children_map[child.uid] = new_obj.get_data(child.name)[0]

        return new_obj, ui_children_map

    def _copy_cherry_pick_objects(
        self,
        parent,
        clear_cache: bool,
    ) -> dict[UUID, ObjectBase | Data | PropertyGroup]:
        """
        Copy the objects and return a map of the copied objects and children.

        :param parent: The parent of the copied objects.
        :param clear_cache: Indicate whether to clear the cache.

        :return: A map of the copied objects and children.
        """
        ui_objects_map: dict[UUID, ObjectBase] = {}
        ui_children_map: dict[UUID, Data | PropertyGroup] = {}

        if self._uijson_objects is not None:
            cherry_pick = (
                [child.uid for child in self._uijson_children]
                if self._uijson_children
                else []
            )
            for obj in self._uijson_objects:
                new_obj, obj_children = self._copy_cherry_pick_object(
                    obj, parent=parent, clear_cache=clear_cache, cherry_pick=cherry_pick
                )
                ui_objects_map[obj.uid] = new_obj
                ui_children_map = {**ui_children_map, **obj_children}

        return {**ui_objects_map, **ui_children_map}

    def _get_copy_entity_mapper(
        self,
        parent,
        copy_relatives: bool,
        clear_cache: bool,
    ) -> dict[UUID, ObjectBase | Data | PropertyGroup | Group]:
        """
        Copy the entities and get a map of the copied entities.

        The function copy the entities contains in the group
        and return a map of the copied entities.
        If the parent is the same, no copy is done and an empty map is returned.
        For objects, only the ones referenced in the options are copied.
        For groups, all the groups are copied.

        :param parent: The parent of the copied entities.
        :param copy_relatives: If the relatives should be copied.
        :param clear_cache: Indicate whether to clear the cache.

        :return: A map of the copied entities.
        """
        new_object_mapper: dict[UUID, ObjectBase | Data | PropertyGroup | Group] = {}
        if copy_relatives and (parent is not None or parent != self.parent):
            if self._uijson_objects is not None:
                new_object_mapper.update(
                    self._copy_cherry_pick_objects(
                        parent=parent, clear_cache=clear_cache
                    )
                )

            if self._uijson_groups is not None:
                # for the group, copying everything because it's too complicated
                for group in self._uijson_groups:
                    new_group = group.copy(
                        parent=parent, copy_children=True, clear_cache=clear_cache
                    )
                    new_object_mapper[group.uid] = new_group

        return new_object_mapper

    def _replace_uuids_in_options(
        self,
        entity_map: dict[UUID, ObjectBase | Data | PropertyGroup | Group],
        parent,
    ) -> dict | None:
        """
        Replace uids in options with the copied entities.

        :param entity_map: The map of the copied entities.

        :return: Either None if options is None, or the options with replaced uids.
        """
        if parent is None or parent == self.parent or not entity_map:
            return self.options

        def replace_uuids(val: Any):
            """
            Replace uids in options with the copied entities.

            :param val: the value to replace uids in

            :return: The copied entity if val is a uid of a copied entity, else returns val
            """
            if isinstance(val, UUID) and val in entity_map:
                return entity_map[val].uid
            return val

        if self.options:
            return dict_mapper(self.options, [entity2uuid, replace_uuids])
        return None

    def copy(
        self,
        parent=None,
        *,
        copy_children: bool = False,
        copy_relatives: bool = True,
        clear_cache: bool = False,
        **kwargs,
    ) -> UIJsonGroup | None:
        """
        Sub-class extension of :func:`~geoh5py.groups.base.Group.copy`.

        :param parent: The parent to copy the entity to.
        :param copy_children: Whether to copy the children of the entity.
        :param copy_relatives: If true, the objects and groups referenced in the options are copied.
        :param clear_cache: Indicate whether to clear the cache.
        :param kwargs: other keyword arguments.

        :return: The copied entity.
        """
        omit_list = self._omit_list.copy()
        if "omit_list" in kwargs:
            omit_list.extend(kwargs.pop("omit_list"))

        entity_map = self._get_copy_entity_mapper(parent, copy_relatives, clear_cache)

        copied = super().copy(
            parent=parent,
            copy_children=copy_children,
            clear_cache=clear_cache,
            omit_list=omit_list,
        )

        copied.options = self._replace_uuids_in_options(entity_map, parent)

        return copied

    # todo: do the same for copy from extent?

    @property
    def options(self) -> dict:
        """
        Metadata attached to the entity.
        """
        return self._options

    @options.setter
    def options(self, value: dict):
        def extract_entities(val: Any):
            """
            Extract entities from options.

            It saves the entities in the corresponding lists and returns their uuid.

            :param val: the value to extract entities from

            :return: The uuid of the entity if val is an entity, else returns val
            """

            val = uuid2entity(val, self.workspace)

            if isinstance(val, ObjectBase):
                if self._uijson_objects is None:
                    self._uijson_objects = []
                if val not in self._uijson_objects:
                    self._uijson_objects.append(val)
                return val.uid
            if isinstance(val, Group):
                if self._uijson_groups is None:
                    self._uijson_groups = []
                if val not in self._uijson_groups:
                    self._uijson_groups.append(val)
                return val.uid
            if isinstance(val, Data | PropertyGroup):
                if self._uijson_children is None:
                    self._uijson_children = []
                if val not in self._uijson_children:
                    self._uijson_children.append(val)
                return val.uid
            return val

        if not isinstance(value, dict):
            raise ValueError(f"Input 'options' must be of type {dict}.")

        self._options = dict_mapper(value, [str2uuid, extract_entities])

        if self.on_file:
            self.workspace.update_attribute(self, "options")

    def add_ui_json(self, name: str | None = None):
        """
        Add ui.json file to entity.

        :param name: Name of the file in the workspace.
        """
        if self.options is None:
            raise ValueError("UIJsonGroup must have options set.")

        json_str = json.dumps(stringify(self.options), indent=4)
        bytes_data = json_str.encode("utf-8")

        if name is None:
            name = self.name

        self.add_file(bytes_data, name=f"{name}.ui.json")

    @staticmethod
    def format_input_options(options: dict | np.ndarray | bytes | None) -> dict:
        """
        Format input options to a dictionary.

        :param options: Input options.

        :return: Formatted options.
        """

        if options is None:
            return {}

        if isinstance(options, np.ndarray):
            options = options[0]

        if isinstance(options, bytes):
            options = str_json_to_dict(options)

        if not isinstance(options, dict):
            raise ValueError(f"Input 'options' must be of type {dict}.")

        return options
