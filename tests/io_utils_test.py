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

from geoh5py.workspace import Workspace
from geoh5py.objects import Points
from geoh5py.groups import ContainerGroup
from geoh5py.io.utils import (
    is_uuid, entity2uuid, uuid2entity, str2uuid,
    as_str_if_uuid, bool_value, str_from_utf8_bytes
)


from uuid import uuid4
import numpy as np
import os

def test_is_uuid():
    assert is_uuid(uuid4())
    assert not is_uuid("sldkfj")
    assert not is_uuid(3)

def test_entity2uuid(tmp_path):
    ws = Workspace(os.path.join(tmp_path, "test.geoh5"))
    xyz = np.array([[1,2,3],[4,5,6]])
    points = Points.create(ws, vertices=xyz, name="test_points")
    group = ContainerGroup.create(ws)
    assert is_uuid(entity2uuid(points))
    assert is_uuid(entity2uuid(group))

def test_uuid2entity(tmp_path):
    ws = Workspace(os.path.join(tmp_path, "test.geoh5"))
    xyz = np.array([[1,2,3],[4,5,6]])
    points = Points.create(ws, vertices=xyz, name="test_points")
    assert points.uid == uuid2entity(points.uid, ws).uid
    assert uuid2entity(uuid4(), ws) is None
    assert uuid2entity(5, ws) == 5

def test_str2uuid():
    test_uuid = uuid4()
    assert str2uuid(str(test_uuid)) == test_uuid
    assert str2uuid(5) == 5

def test_as_str_if_uuid():
    test_uuid = uuid4()
    assert as_str_if_uuid(test_uuid).lstrip('{').rstrip('}') == str(test_uuid)
    assert as_str_if_uuid(5) == 5

def test_bool_value():
    assert bool_value(1)
    assert not bool_value(0)

def test_str_from_utf8_bytes():
    assert str_from_utf8_bytes('s'.encode("utf8")) == 's'