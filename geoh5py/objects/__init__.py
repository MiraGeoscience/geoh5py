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


# pylint: disable=unused-import, cyclic-import
# flake8: noqa

from .block_model import BlockModel
from .cell_object import CellObject
from .curve import Curve
from .drape_model import DrapeModel
from .drillhole import Drillhole
from .geo_image import GeoImage
from .grid2d import Grid2D
from .integrator import IntegratorPoints, NeighbourhoodSurface
from .label import Label
from .notype_object import NoTypeObject
from .object_base import ObjectBase
from .object_type import ObjectType
from .octree import Octree
from .points import Points
from .slicer import Slicer
from .surface import Surface
from .surveys.direct_current import CurrentElectrode, PotentialElectrode
from .surveys.electromagnetics.airborne_fem import (
    AirborneFEMReceivers,
    AirborneFEMTransmitters,
)
from .surveys.electromagnetics.airborne_tem import (
    AirborneTEMReceivers,
    AirborneTEMTransmitters,
)
from .surveys.electromagnetics.base import (
    AirborneEMSurvey,
    FEMSurvey,
    LargeLoopGroundEMSurvey,
    MovingLoopGroundEMSurvey,
    TEMSurvey,
)
from .surveys.electromagnetics.ground_fem import (
    LargeLoopGroundFEMReceivers,
    LargeLoopGroundFEMTransmitters,
    MovingLoopGroundFEMReceivers,
    MovingLoopGroundFEMTransmitters,
)
from .surveys.electromagnetics.ground_tem import (
    LargeLoopGroundTEMReceivers,
    LargeLoopGroundTEMTransmitters,
    MovingLoopGroundTEMReceivers,
    MovingLoopGroundTEMTransmitters,
)
from .surveys.electromagnetics.magnetotellurics import MTReceivers
from .surveys.electromagnetics.tipper import TipperBaseStations, TipperReceivers
from .surveys.magnetics import AirborneMagnetics
from .text import TextObject
from .vp_model import VPModel

GA_STRING_TO_OBJECT: dict[str, type[ObjectBase]] = {
    "Block Model": BlockModel,
    "Curve": Curve,
    "Draped Model": DrapeModel,
    "Drillhole": Drillhole,
    "GeoImage": GeoImage,
    "2D grid": Grid2D,
    "Label": Label,
    "Octree Grid": Octree,
    "Points": Points,
    "Surface": Surface,
    "CurrentElectrode": CurrentElectrode,
    "PotentialElectrode": PotentialElectrode,
    "Survey Airborne FEM Rx": AirborneFEMReceivers,
    "Survey Airborne FEM Tx": AirborneFEMTransmitters,
    "Survey Airborne TEM Rx": AirborneTEMReceivers,
    "Survey Airborne TEM Tx": AirborneTEMTransmitters,
    "Survey FEM rx (arbitrary wire)": LargeLoopGroundFEMReceivers,
    "Survey FEM tx (arbitrary wire))": LargeLoopGroundFEMTransmitters,
    "Survey Ground FEM Rx": MovingLoopGroundFEMReceivers,
    "Survey Ground FEM Tx": MovingLoopGroundFEMTransmitters,
    "Ground TEM Rx (large-loop)": LargeLoopGroundTEMReceivers,
    "Ground TEM Tx (large-loop)": LargeLoopGroundTEMTransmitters,
    "Survey Ground TEM Rx": MovingLoopGroundTEMReceivers,
    "Survey Ground TEM Tx": MovingLoopGroundTEMTransmitters,
    "Survey Airborne MT Rx": MTReceivers,
    "Survey ZTEM Rx": TipperReceivers,
    "Survey ZTEM Base Stations": TipperBaseStations,
    "Survey Airborne Magnetics": AirborneMagnetics,
    "Text Object": TextObject,
    "VP Model": VPModel,
}
