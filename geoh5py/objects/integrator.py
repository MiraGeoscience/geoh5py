#  Copyright (c) 2024 Mira Geoscience Ltd.
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

import uuid

from .points import Points
from .surface import Surface


class IntegratorPoints(Points):
    """
    INTEGRATOR Points object.
    Sub-class of :obj:`geoh5py.objects.points.Points`.
    """

    _TYPE_UID = uuid.UUID("{6832ACF3-78AA-44D3-8506-9574A3510C44}")
    _default_name = "Geoscience INTEGRATOR Points"


class NeighbourhoodSurface(Surface):
    """
    Points object made up of vertices.
    """

    _TYPE_UID = uuid.UUID("{88087FB8-76AE-445B-9CDF-68DBCE530404}")
    _default_name = "Neighbourhood Surface"
