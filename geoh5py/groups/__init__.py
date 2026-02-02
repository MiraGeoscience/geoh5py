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


# pylint: disable=unused-import
# flake8: noqa

from .base import Group
from .container import ContainerGroup
from .custom import CustomGroup
from .drillhole import DrillholeGroup, IntegratorDrillholeGroup
from .giftools import GiftoolsGroup
from .group_type import GroupType
from .integrator import (
    AirborneTheme,
    BlastingDataSet,
    BlastingTheme,
    EarthModelsTheme,
    GeochemistryMineralogyDataSet,
    GeochemistryMineralogyTheme,
    GeophysicsTheme,
    GroundTheme,
    IncidentDataSet,
    IncidentsTheme,
    IntegratorGroup,
    IntegratorNeighbourhoods,
    IntegratorMaps,
    IntegratorMapFile,
    IntegratorProject,
    MineProductionTheme,
    ObservationPointsTheme,
    QueryGroup,
    RockPropertiesTheme,
    SamplesTheme,
)
from .interpretation_section import InterpretationSection
from .notype import NoTypeGroup
from .property_group import PropertyGroup, GroupTypeEnum
from .root import RootGroup
from .simpeg import SimPEGGroup
from .survey import AirborneGeophysics
from .uijson import UIJsonGroup

GA_STRING_TO_GROUP: dict[str, type[Group]] = {
    "Container": ContainerGroup,
    "Drillholes": DrillholeGroup,
    "Drillholes & Wells Data Set": IntegratorDrillholeGroup,
    "GIFtools Project": GiftoolsGroup,
    "Airborne Theme": AirborneTheme,
    "Blasting Data Set": BlastingDataSet,
    "Blasting Theme": BlastingTheme,
    "Earth Models Theme": EarthModelsTheme,
    "Geochemistry & Mineralogy Data Set": GeochemistryMineralogyDataSet,
    "Geochemistry & Mineralogy Theme": GeochemistryMineralogyTheme,
    "Geophysics Theme": GeophysicsTheme,
    "Ground Theme": GroundTheme,
    "Incident Data Set": IncidentDataSet,
    "Incidents Theme": IncidentsTheme,
    "Geoscience INTEGRATOR": IntegratorGroup,
    "Neighbourhoods": IntegratorNeighbourhoods,
    "Geoscience INTEGRATOR Map File Group": IntegratorMaps,
    "Geoscience INTEGRATOR Map File": IntegratorMapFile,
    "Geoscience INTEGRATOR Project": IntegratorProject,
    "Mine Production Theme": MineProductionTheme,
    "Observation Points Theme": ObservationPointsTheme,
    "Rock Properties Theme": RockPropertiesTheme,
    "Samples Theme": SamplesTheme,
    "Interpretation": InterpretationSection,
    "NoType": NoTypeGroup,
    "SimPEG": SimPEGGroup,
    "airborne": AirborneGeophysics,
    "Python plugin": UIJsonGroup,
}
