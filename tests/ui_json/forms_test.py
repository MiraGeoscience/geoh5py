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

import uuid

import numpy as np
import pytest
from pydantic import ValidationError

from geoh5py import Workspace
from geoh5py.objects import Curve, Points, Surface
from geoh5py.ui_json.forms import (
    Association,
    BaseForm,
    BoolForm,
    ChoiceForm,
    DataForm,
    DataType,
    FileForm,
    FloatForm,
    IntegerForm,
    MultiChoiceForm,
    ObjectForm,
    StringForm,
)


def test_base_form():
    form = BaseForm(label="name", value="test")
    assert form.label == "name"
    assert form.value == "test"
    assert form.model_fields_set == {"label", "value"}


def test_base_form_config_extra():
    form = BaseForm(label="name", value="test", extra="stuff")
    assert form.extra == "stuff"
    assert form.model_extra == {"extra": "stuff"}


def test_base_form_config_frozen():
    form = BaseForm(label="name", value="test")
    with pytest.raises(ValidationError, match="Instance is frozen"):
        form.label = "new"


def test_base_form_config_alias():
    form = BaseForm(
        label="name",
        value="test",
        dependency="my_param",
        group_optional=True,
        dependencyType="enabled",
    )
    assert form.group_optional
    assert form.dependency_type == "enabled"
    assert not hasattr(form, "dependencyType")

    with pytest.raises(ValidationError, match="dependencyType"):
        form = BaseForm(
            label="name", value="test", dependency="my_param", dependencyType=1
        )


def test_dependency_type_enum():
    form = BaseForm(
        label="name", value="test", dependency="my_param", dependency_type="enabled"
    )
    assert form.dependency_type == "enabled"

    with pytest.raises(
        ValidationError, match="Input should be 'enabled' or 'disabled'"
    ):
        _ = BaseForm(
            label="name", value="test", dependency="my_param", dependency_type="invalid"
        )


def test_base_form_serieralization():
    form = BaseForm(label="name", value="test", extra="stuff")
    json = form.json_string
    assert all(k in json for k in ["label", "value", "extra"])
    form = BaseForm(label="name", value="test", dependency_type="disabled")
    json = form.json_string
    assert "dependencyType" in json


def test_string_form():
    form = StringForm(label="name", value="test")
    assert form.label == "name"
    assert form.value == "test"
    msg = "Input should be a valid string"
    with pytest.raises(ValueError, match=msg):
        _ = StringForm(label="name", value=1)


def test_bool_form():
    form = BoolForm(label="name", value=True)
    assert form.label == "name"
    assert form.value
    msg = "Input should be a valid boolean"
    with pytest.raises(ValueError, match=msg):
        _ = BoolForm(label="name", value="nope")


def test_integer_form():
    form = IntegerForm(label="name", value=2)
    assert form.label == "name"
    assert form.value == 2
    assert form.min == -np.inf
    assert form.max == np.inf
    msg = "Input should be a valid integer"
    with pytest.raises(ValueError, match=msg):
        _ = IntegerForm(label="name", value="nope")


def test_float_form():
    form = FloatForm(label="name", value=2.0)
    assert form.label == "name"
    assert form.value == 2.0
    assert form.min == -np.inf
    assert form.max == np.inf
    assert form.precision == 2
    assert form.line_edit


def test_choice_form():
    form = ChoiceForm(label="name", value="test", choice_list=["test"])
    assert form.label == "name"
    assert form.value == "test"
    assert form.choice_list == ["test"]
    assert '"value":"test"' in form.json_string
    msg = r"Provided value: 'nope' is not a valid choice"
    with pytest.raises(ValidationError, match=msg):
        _ = ChoiceForm(label="name", value="nope", choice_list=["test"])


def test_multi_choice_form():
    form = MultiChoiceForm(
        label="names", value=["test", "other"], choice_list=["test", "other", "another"]
    )
    assert form.value == ["test", "other"]
    assert form.choice_list == ["test", "other", "another"]
    assert '"value":["test","other"]' in form.json_string

    form = MultiChoiceForm(label="names", value="test", choice_list=["test", "other"])
    assert form.value == ["test"]
    assert '"value":["test"]' in form.json_string


def test_file_form(tmp_path):
    paths = [tmp_path / "my_file.ext", tmp_path / "my_other_file.ext"]
    _ = [p.touch() for p in paths]
    form = FileForm(
        label="file",
        value=str(paths[0]),
        file_description=["something"],
        file_type=["ext"],
    )
    assert form.label == "file"
    assert form.value == [paths[0]]
    assert form.file_description == ["something"]
    assert form.file_type == ["ext"]
    assert not form.file_multi
    assert 'my_file.ext",' in form.json_string

    form = FileForm(
        label="file",
        value=";".join([str(p) for p in paths]),
        file_description=["something"],
        file_type=["ext"],
        file_multi=True,
    )
    assert form.label == "file"
    assert form.value == paths
    assert form.file_description == ["something"]
    assert form.file_type == ["ext"]
    assert form.file_multi

    msg = "does not exist"
    with pytest.raises(ValidationError, match=msg):
        _ = FileForm(
            label="file",
            value=str(tmp_path / "not_a_file.ext"),
            file_description=["something"],
            file_type=["ext"],
        )

    msg = "File description and type lists must be the same length"
    with pytest.raises(ValidationError, match=msg):
        _ = FileForm(
            label="file",
            value=str(paths[0]),
            file_description=["something", "else"],
            file_type=["ext"],
        )

    msg = "have invalid extensions"
    with pytest.raises(ValidationError, match=msg):
        _ = FileForm(
            label="file",
            value=str(paths[0]),
            file_description=["something"],
            file_type=["doc"],
        )


def test_object_form():
    obj_uid = str(uuid.uuid4())
    form = ObjectForm(label="name", value=obj_uid, mesh_type=[Points, Surface])
    assert form.label == "name"
    assert form.value == uuid.UUID(obj_uid)
    assert form.mesh_type == [Points, Surface]

    with pytest.raises(ValidationError, match="Input should be a valid UUID"):
        _ = ObjectForm(
            label="name",
            value="not a uuid",
            mesh_type=[Points, Surface],
        )

    bad_uid_string = str(uuid.uuid4())
    msg = (
        f"Provided type_uid string {bad_uid_string} is not a recognized geoh5py "
        f"object or group type uid"
    )
    with pytest.raises(ValidationError, match=msg):
        _ = ObjectForm(label="name", value=obj_uid, mesh_type=[Points, bad_uid_string])


def test_object_form_mesh_type():
    obj_uid = str(uuid.uuid4())
    form = ObjectForm(label="name", value=obj_uid, mesh_type=Points)
    assert form.mesh_type == [Points]

    obj_uid = str(uuid.uuid4())
    form = ObjectForm(label="name", value=obj_uid, mesh_type=Points.default_type_uid())
    assert form.mesh_type == [Points]

    obj_uid = str(uuid.uuid4())
    form = ObjectForm(
        label="name", value=obj_uid, mesh_type=str(Points.default_type_uid())
    )
    assert form.mesh_type == [Points]

    obj_uid = str(uuid.uuid4())
    form = ObjectForm(
        label="name", value=obj_uid, mesh_type=[Points, str(Curve.default_type_uid())]
    )
    assert form.mesh_type == [Points, Curve]


def test_object_form_mesh_type_as_classes(tmp_path):
    ws = Workspace(tmp_path / "test.geoh5")
    points = Points.create(ws, vertices=np.random.rand(10, 3))

    form = ObjectForm(
        label="name",
        value=points.uid,
        mesh_type=[str(Points.default_type_uid()), str(Curve.default_type_uid())],
    )

    assert isinstance(ws.get_entity(form.value)[0], tuple(form.mesh_type))


def test_object_form_empty_string_handling():
    form = ObjectForm(label="name", value="", mesh_type=[Points, Surface])
    assert not form.value


def test_data_form():
    data_uid = str(uuid.uuid4())
    form = DataForm(
        label="name",
        value=data_uid,
        parent="my_param",
        association="Vertex",
        data_type="Float",
    )
    assert form.label == "name"
    assert form.value == uuid.UUID(data_uid)
    assert form.parent == "my_param"
    assert form.association == "Vertex"
    assert form.data_type == "Float"

    form = DataForm(
        label="name",
        value=data_uid,
        parent="Da-da",
        association=["Vertex", "Cell"],
        data_type=["Float", "Integer"],
    )
    assert form.association == [Association.VERTEX, Association.CELL]
    assert form.data_type == [DataType.FLOAT, DataType.INTEGER]

    with pytest.raises(
        ValidationError, match="Value must be numeric if is_value is True."
    ):
        _ = DataForm(
            label="name",
            value=data_uid,
            parent="my_param",
            association="Vertex",
            data_type="Float",
            is_value=True,
        )
    with pytest.raises(
        ValidationError, match="A property must be provided if is_value is used"
    ):
        _ = DataForm(
            label="name",
            value=1.0,
            parent="my_param",
            association="Vertex",
            data_type="Float",
            is_value=False,
        )


def test_flatten():
    param = BaseForm(label="my_param", value=2)
    assert param.flatten() == 2

    data_uid = str(uuid.uuid4())
    form = DataForm(
        label="name",
        value=data_uid,
        parent="my_param",
        association="Vertex",
        data_type="Float",
    )
    assert str(form.flatten()) == data_uid

    form = DataForm(
        label="name",
        value=0.0,
        parent="my_param",
        association="Vertex",
        data_type="Float",
        property="",
        is_value=True,
    )
    assert form.flatten() == 0.0

    form = DataForm(
        label="name",
        value=0.0,
        parent="my_param",
        association="Vertex",
        data_type="Float",
        property=data_uid,
        is_value=False,
    )

    assert str(form.flatten()) == data_uid
