# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                     '
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


# pylint: disable=duplicate-code

from __future__ import annotations

import random
import string

import numpy as np
import pytest
from h5py import special_dtype

from geoh5py.data import BooleanData, FloatData, ReferencedData
from geoh5py.objects import Drillhole
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_drillhole_data(tmp_path):
    h5file_path = tmp_path / r"testCurve.geoh5"
    well_name = "bullseye"
    n_data = 10

    with Workspace(version=1.0).save_as(h5file_path) as workspace:
        # Create a workspace
        max_depth = 100

        with pytest.warns(UserWarning, match="No 'collar' provided"):
            well = Drillhole.create(
                workspace,
                name=well_name,
                default_collocation_distance=1e-5,
            )

        well.collar = [0.0, 10.0, 10]
        well.surveys = np.c_[
            np.linspace(0, max_depth, n_data),
            np.ones(n_data) * 45.0,
            np.linspace(-89, -75, n_data),
        ]

        with pytest.raises(ValueError, match="Origin must be a list or numpy array"):
            well.collar = [1.0, 10]

        value_map = {}
        for ref in range(8):
            value_map[ref] = "".join(
                random.choice(string.ascii_lowercase) for i in range(8)
            )
        value_map[0] = "Unknown"

        # Create random from-to
        from_to_a = np.sort(
            np.random.uniform(low=0.05, high=max_depth, size=(50,))
        ).reshape((-1, 2))
        from_to_b = np.vstack([from_to_a[0, :], [30.1, 55.5], [56.5, 80.2]])

        with pytest.raises(ValueError) as error:
            well.add_data(
                {
                    "interval_values": {
                        "values": np.random.randn(from_to_a.shape[0]),
                        "from-to": from_to_a[1:, 0],
                    },
                }
            )

        assert "Mismatch between input" in str(error)

        with pytest.raises(ValueError) as error:
            well.add_data(
                {
                    "interval_values": {
                        "values": np.random.randn(from_to_a.shape[0]),
                        "from-to": from_to_a[:, 0],
                    },
                }
            )

        assert "The `from-to` values must have shape(*, 2)." in str(error)

        with pytest.raises(UserWarning) as error:
            well.add_data(
                {
                    "interval_values": {
                        "values": np.random.randn(from_to_a.shape[0]),
                        "from-to": from_to_a[:, 0],
                        "collocation_distance": -1,
                    },
                }
            )

        assert "Input depth 'collocation_distance' must be >0." in str(error)

        # Add from-to data
        data_objects = well.add_data(
            {
                "interval_values": {
                    "values": np.random.randn(from_to_a.shape[0]),
                    "from-to": from_to_a.tolist(),
                },
                "int_interval_list": {
                    "values": [1, 2, 3],
                    "from-to": from_to_b,
                    "value_map": {1: "Unit_A", 2: "Unit_B", 3: "Unit_C"},
                    "type": "referenced",
                },
                "text_list": {
                    "values": np.array(
                        [
                            "".join(
                                random.choice(string.ascii_lowercase) for _ in range(6)
                            )
                            for _ in range(3)
                        ]
                    ),
                    "from-to": from_to_b,
                    "type": "TEXT",
                },
            }
        )

        assert well.n_cells == (from_to_a.shape[0] + 2), (
            "Error with number of cells on interval data creation."
        )
        assert well.n_vertices == (from_to_a.size + 4), (
            "Error with number of vertices on interval data creation."
        )
        assert not np.any(np.isnan(well.get_data("FROM")[0].values)), (
            "FROM values not fully set."
        )
        assert not np.any(np.isnan(well.get_data("TO")[0].values)), (
            "TO values not fully set."
        )
        assert (
            well.get_data("TO")[0].values.shape[0]
            == well.get_data("FROM")[0].values.shape[0]
            == well.n_cells
        ), "Shape or FROM to n_cells differ."

        with pytest.raises(ValueError) as error:
            well.add_data(
                {
                    "log_values": {
                        "values": np.random.randn(n_data - 1),
                        "depth": np.random.randn(n_data),
                    },
                }
            )

        assert "Mismatch between input 'depth'" in str(error)

        # Add log-data
        data_objects += [
            well.add_data(
                {
                    "log_int": {
                        "depth": np.sort(np.random.rand(n_data) * max_depth),
                        "type": "referenced",
                        "values": np.random.randint(1, high=8, size=n_data),
                        "value_map": value_map,
                    }
                }
            )
        ]

        assert isinstance(data_objects[-1], ReferencedData)

        well.add_data(
            {
                "label": {
                    "association": "OBJECT",
                    "values": "ABC",
                },
            }
        )
        new_count = from_to_a.size + 4 + n_data
        assert well.n_vertices == (new_count), (
            "Error with new number of vertices on log data creation."
        )

        # Add other data for tests
        data_objects += [
            well.add_data(
                {
                    "log_bool": {
                        "depth": np.sort(np.random.rand(n_data) * max_depth),
                        "type": "boolean",
                        "values": np.random.choice([True, False], size=n_data),
                    }
                }
            )
        ]

        assert isinstance(data_objects[-1], BooleanData)

        # Add log-data
        data_objects += [
            well.add_data(
                {
                    "log_float": {
                        "depth": np.sort(np.random.rand(n_data) * max_depth),
                        "type": "FLOAT",
                        "values": np.random.rand(n_data).astype(float),
                    }
                }
            )
        ]

        assert isinstance(data_objects[-1], FloatData)

        # Re-open the workspace and read data back in
        new_workspace = Workspace(h5file_path, version=1.0)
        # Check entities
        compare_entities(
            well,
            new_workspace.get_entity(well_name)[0],
            ignore=["_default_collocation_distance", "_parent"],
        )
        # Force refresh of vector length
        data_objects[0]._values = None
        compare_entities(
            data_objects[0],
            new_workspace.get_entity("interval_values")[0],
            ignore=["_parent"],
        )
        compare_entities(
            data_objects[1],
            new_workspace.get_entity("int_interval_list")[0],
            ignore=["_parent"],
        )
        compare_entities(
            data_objects[3],
            new_workspace.get_entity("log_int")[0],
            ignore=["_parent"],
        )
        compare_entities(
            data_objects[2],
            new_workspace.get_entity("text_list")[0],
            ignore=["_parent"],
        )


def test_no_survey(tmp_path):
    collar = np.r_[0.0, 10.0, 10.0]
    h5file_path = tmp_path / r"testCurve.geoh5"
    with Workspace(version=1.0).save_as(h5file_path) as workspace:
        well = Drillhole.create(workspace, collar=collar)
        depths = [0.0, 1.0, 1000.0]
        locations = well.desurvey(depths)
        solution = np.kron(collar, np.ones((3, 1)))
        solution[:, 2] -= depths

        np.testing.assert_array_almost_equal(locations, solution, decimal=1)


def test_single_survey(tmp_path):
    # Create a simple well
    dist = np.random.rand(1) * 100.0
    azm = np.random.randn(1) * 180.0
    dip = np.random.randn(1) * 180.0

    collar = np.r_[0.0, 10.0, 10.0]
    h5file_path = tmp_path / r"testCurve.geoh5"
    with Workspace(version=1.0).save_as(h5file_path) as workspace:
        well = Drillhole.create(workspace, collar=collar, surveys=np.c_[dist, azm, dip])
        depths = [0.0, 1.0, 1000.0]
        locations = well.desurvey(depths)
        solution = (
            collar[None, :]
            + np.c_[
                depths
                * np.cos(np.deg2rad(450.0 - azm % 360.0))
                * np.cos(np.deg2rad(dip)),
                depths
                * np.sin(np.deg2rad(450.0 - azm % 360.0))
                * np.cos(np.deg2rad(dip)),
                depths * np.sin(np.deg2rad(dip)),
            ]
        )

        np.testing.assert_array_almost_equal(locations, solution, decimal=1)


def test_survey_with_info(tmp_path):
    # Create a simple well
    dist = np.random.rand(3) * 100.0
    azm = np.random.randn(3) * 180.0
    dip = np.random.randn(3) * 180.0
    surveys = np.c_[dist, azm, dip].T.tolist()
    surveys += [["A", "B", "C"]]
    surveys = np.core.records.fromarrays(
        surveys,
        dtype=[
            ("Depth", "<f4"),
            ("Azimuth", "<f4"),
            ("Dip", "<f4"),
            ("Info", special_dtype(vlen=str)),
        ],
    )

    collar = np.r_[0.0, 10.0, 10.0]
    h5file_path = tmp_path / f"{__name__}.geoh5"
    with Workspace(version=1.0).save_as(h5file_path) as workspace:
        Drillhole.create(workspace, name="Han Solo", collar=collar, surveys=surveys)

    with Workspace(h5file_path) as workspace:
        well = workspace.get_entity("Han Solo")[0]
        assert len(well._surveys.dtype) == 4


def test_outside_survey(tmp_path):
    # Create a simple well
    dist = np.random.rand(2) * 100.0
    azm = [np.random.randn(1) * 180.0] * 2
    dip = [np.random.randn(1) * 180.0] * 2

    collar = np.r_[0.0, 10.0, 10.0]
    h5file_path = tmp_path / r"testCurve.geoh5"
    with Workspace(version=1.0).save_as(h5file_path) as workspace:
        well = Drillhole.create(workspace, collar=collar, surveys=np.c_[dist, azm, dip])
        depths = [0.0, 1000.0]
        locations = well.desurvey(depths)
        solution = (
            collar[None, :]
            + np.c_[
                depths
                * np.cos(np.deg2rad(450.0 - azm[-1] % 360.0))
                * np.cos(np.deg2rad(dip[-1])),
                depths
                * np.sin(np.deg2rad(450.0 - azm[-1] % 360.0))
                * np.cos(np.deg2rad(dip[-1])),
                depths * np.sin(np.deg2rad(dip[-1])),
            ]
        )

        np.testing.assert_array_almost_equal(locations, solution, decimal=1)


def test_insert_drillhole_data(tmp_path):
    well_name = "bullseye"
    n_data = 10
    collocation = 1e-5
    h5file_path = tmp_path / r"testCurve.geoh5"

    with Workspace(version=1.0).save_as(h5file_path) as workspace:
        max_depth = 100
        well = Drillhole.create(
            workspace,
            collar=np.r_[0.0, 10.0, 10],
            surveys=np.c_[
                np.linspace(0, max_depth, n_data),
                np.ones(n_data) * 45.0,
                np.linspace(-89, -75, n_data),
            ],
            name=well_name,
            default_collocation_distance=collocation,
        )
        # Add log-data
        data_object = well.add_data(
            {
                "log_values": {
                    "depth": np.sort(np.random.rand(n_data) * max_depth),
                    "values": np.random.randint(1, high=8, size=n_data),
                }
            }
        )

        # Add more data with single match
        old_depths = well.get_data("DEPTH")[0].values
        insert = np.random.randint(0, high=n_data - 1, size=2)
        new_depths = old_depths[insert]
        new_depths[0] -= 2e-6  # Out of tolerance
        new_depths[1] -= 5e-7  # Within tolerance

        well.add_data(
            {
                "match_depth": {
                    "depth": new_depths,
                    "values": np.random.randint(1, high=8, size=2),
                },
            },
            collocation_distance=1e-6,
        )

        assert well.n_vertices == n_data + 1, (
            "Error adding values with collocated tolerance"
        )
        assert np.isnan(data_object.values[insert[0]]), (
            "Old values not re-sorted properly after insertion"
        )

        assert np.where(well.depths.values == new_depths[0])[0] == insert[0], (
            "Depth insertion error"
        )


def test_mask_drillhole_data(tmp_path):
    h5file_path = tmp_path / r"testCurve.geoh5"

    with Workspace(version=1.0).save_as(h5file_path) as workspace:
        well = Drillhole.create(
            workspace,
            collar=np.r_[0.0, 10.0, 10],
        )

        assert well.mask_by_extent(np.vstack([[100, 100], [101, 101]])) is None
        assert well.mask_by_extent(np.vstack([[-1, 9], [1, 11]])).sum() == 1
