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

from pathlib import Path

from geoh5py.data import DataType, GeometricDataConstants
from geoh5py.workspace import Workspace


def test_xyz_dataype(tmp_path: Path):
    h5file_path = tmp_path / f"{__name__}.geoh5"
    with Workspace.create_geoh5(h5file_path) as workspace:
        x_datatype = DataType.for_x_data(workspace)
        assert x_datatype.uid == GeometricDataConstants.x_datatype_uid()
        assert (
            DataType.find(workspace, GeometricDataConstants.x_datatype_uid())
            is x_datatype
        )
        # make sure another call does no re-create another type
        assert DataType.for_x_data(workspace) is x_datatype

        y_datatype = DataType.for_y_data(workspace)
        assert y_datatype.uid == GeometricDataConstants.y_datatype_uid()
        assert (
            DataType.find(workspace, GeometricDataConstants.y_datatype_uid())
            is y_datatype
        )
        # make sure another call does no re-create another type
        assert DataType.for_y_data(workspace) is y_datatype

        z_datatype = DataType.for_z_data(workspace)
        assert z_datatype.uid == GeometricDataConstants.z_datatype_uid()
        assert (
            DataType.find(workspace, GeometricDataConstants.z_datatype_uid())
            is z_datatype
        )
        # make sure another call does no re-create another type
        assert DataType.for_z_data(workspace) is z_datatype
