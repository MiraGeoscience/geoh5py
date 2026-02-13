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


from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geoh5py.groups import InterpretationSection
from geoh5py.groups.interpretation_section import InterpretationSectionParams
from geoh5py.objects import Curve, Slicer
from geoh5py.workspace import Workspace


def test_create_interpretation_section(tmp_path: Path):
    h5file_path = tmp_path / r"test.geoh5"
    with Workspace.create(h5file_path) as workspace:
        # create initial object
        group = InterpretationSection.create(workspace)
        assert group.parent == workspace.root, (
            "Assigned parent=None should default to Root."
        )

        curve = Curve.create(workspace, name="curve1", vertices=np.random.randn(10, 3))

        section = {
            "Normal X": 0.1,
            "Normal Y": 0.2,
            "Normal Z": 0.3,
            "Position X": 0.4,
            "Position Y": 0.5,
            "Position Z": 0.6,
        }

        slicer = Slicer.create(workspace, name="slicer")

        # testing the removes
        group.remove_interpretation_curve(curve.uid)  # nothing happens
        group.remove_interpretation_section(section)  # nothing happens

        assert curve.parent != group

        group.add_interpretation_section(section)

        assert InterpretationSectionParams(**section) in group.interpretation_sections

        group.add_interpretation_curve(curve.uid)

        assert curve.parent == group

        assert (curve,) == group.interpretation_curves

        group.section_object = slicer.uid

        assert slicer == group.section_object

        assert group.metadata == {
            "Interpretation curves": [curve.uid],
            "Interpretation sections": [section],
            "Section object ID": slicer.uid,
        }

        group.can_add_group = True

    # reopen
    with Workspace(h5file_path) as workspace:
        group = workspace.get_entity(group.uid)[0]

        assert group.metadata == {
            "Interpretation curves": ["{%s}" % str(curve.uid)],
            "Interpretation sections": [section],
            "Section object ID": slicer.uid,
        }

        assert group.can_add_group is True

        assert InterpretationSectionParams(**section) in group.interpretation_sections

        assert curve.uid == group.interpretation_curves[0].uid

        assert slicer.uid == group.section_object.uid

        # add another object
        curve2 = Curve.create(workspace, name="curve2", vertices=np.random.randn(10, 3))

        section2 = {
            "Normal X": 0.7,
            "Normal Y": 0.8,
            "Normal Z": 0.9,
            "Position X": 1.0,
            "Position Y": 1.1,
            "Position Z": 1.2,
        }

        group.add_interpretation_section(section2)
        group.add_interpretation_curve(curve2.uid)

        assert InterpretationSectionParams(**section2) in group.interpretation_sections

        assert curve2.uid == group.interpretation_curves[1].uid

        assert curve2.clipping_ids[0] == slicer.uid

        # remove object
        group.remove_interpretation_curve(curve.uid)
        assert group.interpretation_curves == (curve2,)
        group.remove_interpretation_section(section)
        assert InterpretationSectionParams(**section2) in group.interpretation_sections

        group.remove_interpretation_section(section2)

        assert group.interpretation_sections == ()

        group.section_object = None
        assert group.section_object is None


def test_create_interpretation_section_errors(tmp_path: Path):
    h5file_path = tmp_path / r"test.geoh5"
    with Workspace.create(h5file_path) as workspace:
        group = InterpretationSection.create(workspace)

        with pytest.raises(ValueError, match="Invalid key"):
            group._update_to_metadata("bidon", 666)

        with pytest.raises(TypeError, match="'Can add group' must be"):
            group.can_add_group = 2

        param_test = InterpretationSectionParams(
            **{
                "Normal X": 0.1,
                "Normal Y": 0.2,
                "Normal Z": 0.3,
                "Position X": 0.4,
                "Position Y": 0.5,
                "Position Z": 0.6,
            }
        )

        with pytest.raises(TypeError, match="Interpretation section must be"):
            InterpretationSectionParams.create_section("bidon")

        assert param_test == InterpretationSectionParams.create_section(param_test)

        with pytest.raises(
            TypeError,
            match="The 666 object must be a Slicer object. <class 'int'> provided.",
        ):
            group._verify_object(666, "Slicer")
