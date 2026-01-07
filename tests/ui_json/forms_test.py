# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
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
from geoh5py.groups import GroupTypeEnum, PropertyGroup
from geoh5py.objects import Curve, Points, Surface
from geoh5py.ui_json.forms import (
    Association,
    BaseForm,
    BoolForm,
    ChoiceForm,
    DataForm,
    DataGroupForm,
    DataOrValueForm,
    DataRangeForm,
    DataType,
    FileForm,
    FloatForm,
    GroupForm,
    IntegerForm,
    MultiChoiceForm,
    MultiSelectDataForm,
    ObjectForm,
    RadioLabelForm,
    StringForm,
)
from geoh5py.ui_json.ui_json import BaseUIJson


def setup_from_uijson(workspace, form):
    class MyUIJson(BaseUIJson):
        my_param: type(form)

    uijson = MyUIJson.model_construct(
        version="blahblah",
        title="my title",
        geoh5=workspace.h5file,
        run_command="whatever",
        monitoring_directory="don't care",
        conda_environment="don't have one",
        my_param=form,
    )
    uijson.write(workspace.h5file.parent / "test.ui.json")
    uijson = MyUIJson.read(workspace.h5file.parent / "test.ui.json")
    return uijson.my_param


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
        ValidationError, match="Input should be 'enabled', 'disabled', 'show' or 'hide'"
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


def test_hide_dependency_type(tmp_path):
    with Workspace.create(tmp_path / "test.geoh5") as ws:
        form = StringForm(
            label="name", value="test", dependency="my_param", dependency_type="show"
        )
        form = setup_from_uijson(ws, form)
        assert form.dependency_type == "show"


def test_string_form():
    form = StringForm(label="name", value="test")
    assert form.label == "name"
    assert form.value == "test"
    msg = "Input should be a valid string"
    with pytest.raises(ValueError, match=msg):
        _ = StringForm(label="name", value=1)


def test_radio_label_form():
    form = RadioLabelForm(
        label="model type",
        original_label="conductivity",
        alternate_label="resistivity",
        value="conductivity",
    )
    assert form.label == "model type"
    assert form.original_label == "conductivity"
    assert form.alternate_label == "resistivity"
    assert form.value == "conductivity"


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
    assert form.multi_select
    assert form.value == ["test", "other"]
    assert form.choice_list == ["test", "other", "another"]
    assert '"value":["test","other"]' in form.json_string

    form = MultiChoiceForm(label="names", value="test", choice_list=["test", "other"])
    assert form.value == ["test"]
    assert '"value":["test"]' in form.json_string

    with pytest.raises(ValidationError, match="multi_select: True."):
        _ = MultiChoiceForm(
            label="names",
            value="test",
            choice_list=["test", "other"],
            multi_select=False,
        )


def test_file_form(tmp_path):
    with Workspace.create(tmp_path / "test.geoh5") as geoh5:
        paths = [tmp_path / "my_file.ext", tmp_path / "my_other_file.ext"]
        _ = [p.touch() for p in paths]

        file_form = FileForm(
            label="file",
            value=paths[0],
            file_description=["something"],
            file_type=["ext"],
        )

        file_form = setup_from_uijson(geoh5, file_form)

        assert file_form.label == "file"
        assert file_form.value == [paths[0]]
        assert file_form.file_description == ["something"]
        assert file_form.file_type == ["ext"]
        assert not file_form.file_multi
        assert 'my_file.ext",' in file_form.json_string

        file_form = FileForm(
            label="file",
            value=";".join([str(p) for p in paths]),
            file_description=["something"],
            file_type=["ext"],
            file_multi=True,
        )
        file_form = setup_from_uijson(geoh5, file_form)

        assert file_form.label == "file"
        assert file_form.value == paths
        assert file_form.file_description == ["something"]
        assert file_form.file_type == ["ext"]
        assert file_form.file_multi

        msg = "does not exist"
        with pytest.raises(ValidationError, match=msg):
            form = FileForm.model_construct(
                label="file",
                value=str(tmp_path / "not_a_file.ext"),
                file_description=["something"],
                file_type=["ext"],
            )
            setup_from_uijson(geoh5, form)

        msg = "File description and type lists must be the same length"
        with pytest.raises(ValidationError, match=msg):
            form = FileForm.model_construct(
                label="file",
                value=[paths[0]],
                file_description=["something", "else"],
                file_type=["ext"],
            )
            setup_from_uijson(geoh5, form)

        msg = "have invalid extensions"
        with pytest.raises(ValidationError, match=msg):
            form = FileForm.model_construct(
                label="file",
                value=[paths[0]],
                file_description=["something"],
                file_type=["doc"],
            )
            setup_from_uijson(geoh5, form)


def test_directory_form(tmp_path):
    file_form = FileForm(
        label="working directory",
        file_description=["Directory"],
        file_type=["directory"],
        directory_only=True,
        value=str(tmp_path),
    )
    with Workspace.create(tmp_path / "test.geoh5") as geoh5:
        file_form = setup_from_uijson(geoh5, file_form)
        assert file_form.value[0] == tmp_path

        with pytest.raises(ValidationError, match="File type must be"):
            file_form = FileForm.model_construct(
                label="working directory",
                file_description=["Directory"],
                file_type=["ext"],
                directory_only=True,
                value=[str(tmp_path)],
            )
            file_form = setup_from_uijson(geoh5, file_form)
            assert True

        with pytest.raises(ValidationError, match="File description must be"):
            file_form = FileForm(
                label="working directory",
                file_description=["something else"],
                file_type=["directory"],
                directory_only=True,
                value=[str(tmp_path)],
            )
            file_form = setup_from_uijson(geoh5, file_form)
            assert True


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


def test_data_group_form():
    group_uid = str(uuid.uuid4())
    form = DataGroupForm(
        label="name",
        value=group_uid,
        data_group_type="Strike & dip",
        parent="Da-da",
        association=["Vertex", "Cell"],
        data_type=["Float", "Integer"],
    )
    assert form.label == "name"
    assert form.value == uuid.UUID(group_uid)
    assert form.data_group_type == GroupTypeEnum.STRIKEDIP
    assert form.parent == "Da-da"
    assert form.association == [Association.VERTEX, Association.CELL]
    assert form.data_type == [DataType.FLOAT, DataType.INTEGER]


def test_data_or_value_form():
    data_uid = str(uuid.uuid4())
    form = DataOrValueForm(
        label="name",
        value=0.0,
        parent="my_param",
        association="Vertex",
        data_type="Float",
        is_value=False,
        property=data_uid,
    )
    assert form.label == "name"
    assert form.value == 0.0
    assert form.parent == "my_param"
    assert form.association == "Vertex"
    assert form.data_type == "Float"
    assert not form.is_value
    assert form.property == uuid.UUID(data_uid)

    with pytest.raises(
        ValidationError, match="Value must be numeric if is_value is True."
    ):
        _ = DataOrValueForm(
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
        _ = DataOrValueForm(
            label="name",
            value=1.0,
            parent="my_param",
            association="Vertex",
            data_type="Float",
            is_value=False,
        )


def test_multichoice_data_form():
    data_uid_1 = str(uuid.uuid4())
    data_uid_2 = str(uuid.uuid4())

    form = MultiSelectDataForm(
        label="name",
        value=data_uid_1,
        parent="my_param",
        association="Vertex",
        data_type="Float",
    )
    assert form.label == "name"
    assert form.value == [uuid.UUID(data_uid_1)]
    assert form.parent == "my_param"
    assert form.association == "Vertex"
    assert form.data_type == "Float"

    form = MultiSelectDataForm(
        label="name",
        value=[data_uid_1, data_uid_2],
        parent="my_param",
        association="Vertex",
        data_type="Float",
    )
    assert form.value == [uuid.UUID(data_uid_1), uuid.UUID(data_uid_2)]


def test_multichoice_data_form_serialization():
    data_uid_1 = f"{{{uuid.uuid4()!s}}}"
    data_uid_2 = f"{{{uuid.uuid4()!s}}}"
    form = MultiSelectDataForm(
        label="name",
        value=[data_uid_1, data_uid_2],
        parent="my_param",
        association="Vertex",
        data_type="Float",
    )
    data = form.model_dump()
    assert data["value"] == [data_uid_1, data_uid_2]

    form = MultiSelectDataForm(
        label="name",
        value=data_uid_1,
        parent="my_param",
        association="Vertex",
        data_type="Float",
    )
    data = form.model_dump()
    assert data["value"] == [data_uid_1]

    form = MultiSelectDataForm(
        label="name",
        value=[],
        parent="my_param",
        association="Vertex",
        data_type="Float",
        is_value=False,
        property=[data_uid_1, data_uid_2],
    )
    data = form.model_dump()
    assert data["property"] == [data_uid_1, data_uid_2]
    assert data["value"] == []


def test_data_range_form():
    data_uid = str(uuid.uuid4())
    form = DataRangeForm(
        label="name",
        property=data_uid,
        value=[0.0, 1.0],
        parent="my_param",
        association="Vertex",
        data_type="Float",
        range_label="value range",
    )
    assert form.label == "name"
    assert form.property == uuid.UUID(data_uid)
    assert form.value == [0.0, 1.0]
    assert form.parent == "my_param"
    assert form.association == "Vertex"
    assert form.data_type == "Float"
    assert form.range_label == "value range"


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

    form = DataOrValueForm(
        label="name",
        value=0.0,
        parent="my_param",
        association="Vertex",
        data_type="Float",
        property="",
        is_value=True,
    )
    assert form.flatten() == 0.0

    form = DataOrValueForm(
        label="name",
        value=0.0,
        parent="my_param",
        association="Vertex",
        data_type="Float",
        property=data_uid,
        is_value=False,
    )

    assert str(form.flatten()) == data_uid


def test_base_form_infer(tmp_path):
    form = BaseForm.infer({"label": "test", "value": "test"})
    assert form == StringForm
    form = BaseForm.infer({"label": "test", "value": 1})
    assert form == IntegerForm
    form = BaseForm.infer({"label": "test", "value": 1.0})
    assert form == FloatForm
    form = BaseForm.infer({"label": "test", "value": True})
    assert form == BoolForm
    form = BaseForm.infer(
        {"label": "test", "value": str(uuid.uuid4()), "meshType": [Points]}
    )
    assert form == ObjectForm
    form = BaseForm.infer(
        {"label": "test", "value": str(uuid.uuid4()), "mesh_type": Points}
    )
    assert form == ObjectForm
    form = BaseForm.infer(
        {
            "label": "test",
            "value": str(uuid.uuid4()),
            "parent": "my_param",
            "association": "Vertex",
            "dataType": "Float",
        }
    )
    assert form == DataForm
    form = BaseForm.infer(
        {
            "label": "test",
            "value": [str(uuid.uuid4()), str(uuid.uuid4())],
            "parent": "my_param",
            "association": "Vertex",
            "dataType": "Float",
            "multiSelect": True,
        }
    )
    assert form == MultiSelectDataForm
    form = BaseForm.infer(
        {
            "label": "test",
            "value": str(uuid.uuid4()),
            "parent": "my_param",
            "association": "Vertex",
            "dataType": "Float",
            "isValue": True,
            "property": str(uuid.uuid4()),
        }
    )
    assert form == DataOrValueForm
    form = BaseForm.infer(
        {"label": "test", "groupType": PropertyGroup, "value": str(uuid.uuid4())}
    )
    assert form == GroupForm
    form = BaseForm.infer(
        {
            "label": "test",
            "value": tmp_path,
            "directoryOnly": True,
            "fileType": ["ext"],
            "fileDescription": ["something"],
        }
    )
    assert form == FileForm
    form = BaseForm.infer(
        {"label": "test", "value": "test", "choiceList": ["test", "other"]}
    )
    assert form == ChoiceForm
    form = BaseForm.infer(
        {
            "label": "test",
            "multiSelect": True,
            "value": ["test", "other"],
            "choice_list": ["test", "other", "another"],
        }
    )
    assert form == MultiChoiceForm
    form = BaseForm.infer(
        {
            "label": "test",
            "property": str(uuid.uuid4()),
            "value": [0.0, 1.0],
            "parent": "my_param",
            "association": "Vertex",
            "dataType": "Float",
            "rangeLabel": "value range",
        }
    )
    assert form == DataRangeForm
