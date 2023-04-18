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

import numpy as np

from geoh5py.data.visual_parameters import VisualParameters
from geoh5py.objects import Points
from geoh5py.workspace import Workspace


def test_visual_parameters(tmp_path):
    name = "MyTestPointset"

    # Generate a random cloud of points with reference values
    n_data = 12
    values = """
    <IParameterList Version="1.0">
        <Colour>4288806783</Colour>
        <Transparency>0</Transparency>
        <Lighting>true</Lighting>
        <Showcage>true</Showcage>
        <Showgrid>false</Showgrid>
        <Showaxis>false</Showaxis>
        <Smooth>false</Smooth>
        <Originelevation>0</Originelevation>
        <Deformbydata toggled="false"></Deformbydata>
        <Distancefromplane>100</Distancefromplane>
        <Orientation toggled="false">{
                                      "DataGroup": "",
                                      "ManualWidth": false,
                                      "Scale": false,
                                      "ScaleLog": false,
                                      "Size": 30,
                                      "Symbol": "2D arrow",
                                      "Width": 7\n}\n</Orientation>
        <Contours toggled="false">{
                                   "Anchor Parameter": "0",
                                   "Colour Picker": "4294967040",
                                   "Disable on Incompatible": false,
                                   "Interval Parameter": "0",
                                   "Maximum Contours Parameter": "20",
                                   "Property Picker": ""}
        </Contours>
    </IParameterList>
    """

    h5file_path = tmp_path / r"testTextData.geoh5"

    with Workspace(h5file_path) as workspace:
        with Workspace(tmp_path / r"testTextData_copy.geoh5") as new_workspace:
            points = Points.create(
                workspace,
                vertices=np.random.randn(n_data, 3),
                name=name,
                allow_move=False,
            )

            data = points.add_data(
                {
                    "Visual Parameters": {
                        "type": "text",
                        "values": values,
                        "association": "OBJECT",
                    }
                }
            )

            assert isinstance(data, VisualParameters)
