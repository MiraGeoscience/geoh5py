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

from geoh5py.ui_json.forms import MemberKeys


def test_keys():
    mappable_keys = {
        "groupOptional": "group_optional",
        "dependencyType": "dependency_type",
        "groupDependency": "group_dependency",
        "groupDependencyType": "group_dependency_type",
        "lineEdit": "line_edit",
        "choiceList": "choice_list",
        "fileDescription": "file_description",
        "fileType": "file_type",
        "fileMulti": "file_multi",
        "meshType": "mesh_type",
        "dataType": "data_type",
        "dataGroupType": "data_group_type",
        "isValue": "is_value",
    }
    keys = MemberKeys()
    assert list(keys.map(mappable_keys)) == list(mappable_keys.values())
    inv_mappable_keys = {v: k for k, v in mappable_keys.items()}
    assert list(keys.map(inv_mappable_keys, convention="camel")) == list(mappable_keys)
