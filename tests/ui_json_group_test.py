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

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from geoh5py.groups import UIJsonGroup
from geoh5py.shared.utils import compare_entities
from geoh5py.ui_json import constants, templates
from geoh5py.workspace import Workspace

from .property_group_test import make_example


def test_uijson_group(tmp_path):
    h5file_path = tmp_path / r"testUIJSONGroup.geoh5"

    # Create a workspace with group
    with Workspace.create(h5file_path) as workspace:
        curve, _ = make_example(workspace)

        # prepare a fancy uijson
        uijson = deepcopy(constants.default_ui_json)
        uijson["something"] = templates.float_parameter()
        uijson["curve"] = curve
        uijson["data"] = curve.get_data("Period1")[0]
        uijson["property_group"] = curve.get_property_group("myGroup")[0]

        group = UIJsonGroup.create(workspace, name="test", options=uijson)

        # Copy on new workspace
        with Workspace.create(tmp_path / r"testGroup2.geoh5") as new_workspace:
            group.copy(parent=new_workspace)

        group.copy(omit_list=["_name"])

    # Read back and compare
    with Workspace(h5file_path) as workspace:
        with Workspace(tmp_path / r"testGroup2.geoh5") as new_workspace:
            group = workspace.get_entity("test")[0]
            group_copy = workspace.get_entity("UIJson")[0]  # also testing omit _name

            rec_obj = new_workspace.get_entity(group.uid)[0]

            compare_entities(
                group,
                group_copy,
                ignore=[
                    "_parent",
                ],
            )

            compare_entities(
                group,
                rec_obj,
                ignore=[
                    "_parent",
                ],
            )

            rec_obj.add_ui_json("something")

            assert new_workspace.get_entity("something.ui.json")[0]
            assert Path(group.options["geoh5"]).stem == "testUIJSONGroup"
            assert Path(rec_obj.options["geoh5"]).stem == "testGroup2"
            assert rec_obj.options["out_group"]["value"] == str(group.uid)


def test_uijson_group_copy_relatives(tmp_path):
    h5file_path = tmp_path / r"testUIJSONGroupRelatives.geoh5"

    # Create a workspace with group
    with Workspace.create(h5file_path) as workspace:
        # create an objects, a property group and
        curve, _ = make_example(workspace)

        # prepare a fancy uijson
        uijson = deepcopy(constants.default_ui_json)
        uijson["something"] = templates.float_parameter()
        uijson["curve"] = {"a nested dict": curve}
        uijson["data"] = {
            "another one": {
                "and a list too": [
                    {
                        "keep swimming": curve.get_data("Period1")[0],
                    }
                ]
            },
        }
        uijson["property_group"] = curve.get_property_group("myGroup")[0]
        uijson["group"] = {
            "value": UIJsonGroup.create(workspace, name="dummy"),
            "groupType": UIJsonGroup.default_type_uid(),
        }

        group = UIJsonGroup.create(workspace)
        group.options = uijson

        with Workspace.create(tmp_path / r"testGroup2.geoh5") as new_workspace:
            group.copy(
                parent=new_workspace,
                copy_children=True,
                copy_relatives=True,
                name="copy_uijson",
            )

        # copy on same workspace
        # todo: are we adding a (1) to the groups too?
        group.copy(copy_children=True, copy_relatives=True, name="UIJson_2")

    with Workspace(tmp_path / r"testGroup2.geoh5") as new_workspace:
        rec_obj = new_workspace.get_entity("copy_uijson")[0]
        options = rec_obj.options

        assert options["group"]["groupType"] == UIJsonGroup.default_type_uid()

        assert (
            new_workspace.get_entity(options["curve"]["a nested dict"])[0].name
            == "curve"
        )
        assert (
            new_workspace.get_entity(
                options["data"]["another one"]["and a list too"][0]["keep swimming"]
            )[0].name
            == "Period1"
        )
        assert new_workspace.get_entity(options["property_group"])[0].name == "myGroup"
        # all children are getting copied
        assert new_workspace.get_entity("Period2")[0].name == "Period2"

    with Workspace(h5file_path) as workspace:
        original = workspace.get_entity("UIJson")[0]
        rec_obj = workspace.get_entity("UIJson_2")[0]

        original_options = original.options
        rec_obj_options = rec_obj.options
        original_options.pop("out_group")
        rec_obj_options.pop("out_group")
        assert original_options == rec_obj_options


def test_copy_uijson_group_no_option(tmp_path):
    h5file_path = tmp_path / r"testUIJSONGroupRelatives.geoh5"

    # Create a workspace with group
    with Workspace.create(h5file_path) as workspace:
        group = UIJsonGroup.create(workspace)

        with Workspace.create(tmp_path / r"testGroup2.geoh5") as new_workspace:
            group.copy(
                parent=new_workspace,
                copy_children=True,
                copy_relatives=True,
                name="copy_uijson",
            )
        group.copy(name="UIJson_2", copy_children=True, copy_relatives=True)

        # test editing an option
        group.modify_option("new_option", "bidon")

        assert group.options["new_option"] == "bidon"

    with Workspace(tmp_path / r"testGroup2.geoh5") as new_workspace:
        rec_obj = new_workspace.get_entity("copy_uijson")[0]
        assert len(rec_obj.options) == 0

    with Workspace(h5file_path) as workspace:
        original = workspace.get_entity("UIJson")[0]
        assert original.options["new_option"] == "bidon"

        rec_obj = workspace.get_entity("UIJson_2")[0]
        assert len(rec_obj.options) == 0


def test_uijson_group_errors(tmp_path):
    h5file_path = tmp_path / r"testUIJSONGroupErrors.geoh5"

    # Create a workspace with group
    workspace = Workspace.create(h5file_path)
    group = UIJsonGroup.create(workspace, name="test")

    with pytest.raises(ValueError, match="UIJsonGroup must have options"):
        group.add_ui_json("something")

    with pytest.raises(TypeError, match="Input 'options' must be "):
        group.options = "bidon"

    group.options = group.format_input_options(np.array([b'{"name":"bidon"}']))

    group.add_ui_json()

    assert workspace.get_entity("test.ui.json")[0]

    with pytest.raises(ValueError, match="Cannot modify the 'geoh5' entry"):
        group.modify_option("geoh5", "bidon")


def test_double_uijson_group(tmp_path):
    h5file_path = tmp_path / r"testDoubleUIJSONGroup.geoh5"

    # Create a workspace with group
    with Workspace.create(h5file_path) as workspace:
        curve, _ = make_example(workspace)

        # prepare a fancy uijson
        uijson = deepcopy(constants.default_ui_json)
        uijson["something"] = templates.float_parameter()
        uijson["curve"] = curve

        group_1 = UIJsonGroup.create(workspace, name="test_1", options=uijson)
        group_2 = UIJsonGroup.create(workspace, name="test_2", options=uijson)

        with Workspace() as new_workspace:
            group_1.copy(parent=new_workspace)
            group_2.copy(parent=new_workspace)

            assert len(new_workspace.get_entity("curve")) == 1
