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

import numpy as np

from .grid_object import GridObject
from .object_base import ObjectType


class DrapeModel(GridObject):
    """
    Drape (curtain) model object made up of layers and prisms.
    """

    __TYPE_UID = uuid.UUID("{C94968EA-CF7D-11EB-B8BC-0242AC130003}")
    __LAYERS_DTYPE = np.dtype([("I", "<i4"), ("K", "<i4"), ("Bottom elevation", "<f8")])
    __PRISM_DTYPE = np.dtype(
        [
            ("Top easting", "<f8"),
            ("Top northing", "<f8"),
            ("Top elevation", "<f8"),
            ("First layer", "<i4"),
            ("Layer count", "<i4"),
        ]
    )

    def __init__(
        self,
        object_type: ObjectType,
        layers: np.ndarray | list | tuple = (0, 0, -1.0),
        prisms: np.ndarray | list | tuple = (0.0, 0.0, 0.0, 0, 1),
        **kwargs,
    ):
        self._layers: np.ndarray = self.validate_layers(layers)
        self._prisms: np.ndarray = self.validate_prisms(prisms)

        super().__init__(object_type, layers=layers, prisms=prisms, **kwargs)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @property
    def centroids(self) -> np.ndarray:
        """
        :obj:`numpy.array` of :obj:`float`,
        shape (:obj:`~geoh5py.objects.drape_model.Drapemodel.n_cells`, 3):
        Cell center locations in world coordinates.

        .. code-block:: python

            centroids = [
                [x_1, y_1, z_1],
                ...,
                [x_N, y_N, z_N]
            ]
        """
        if getattr(self, "_centroids", None) is None:
            self._centroids = np.vstack(
                [
                    np.ones((int(val), 3)) * self.prisms[ii, :3]
                    for ii, val in enumerate(self.prisms[:, 4])
                ]
            )
            tops = np.hstack(
                [
                    np.r_[
                        cells[2],
                        self.layers[int(cells[3]) : int(cells[3] + cells[4] - 1), 2],
                    ]
                    for cells in self.prisms.tolist()
                ]
            )
            self._centroids[:, 2] = (tops + self.layers[:, 2]) / 2.0

        return self._centroids

    @property
    def layers(self) -> np.ndarray:
        """
        :obj:`numpy.array`, shape(*, 3): Layers in the drape model with columns: X
        (prism index), K (depth index), elevation (cell bottom)).
        shape(*, 3) organized into blocks representing each prism in the model.

        .. code-block:: python

            layers = [
                [x_1, k_1, z_11],
                [x_1, k_2, z_12],
                ...
                [x_1, k_N, z_1N],
                .
                .
                .
                [x_M, k_1, z_M1],
                [x_M, k_2, z_M2],
                ...
                [x_M, k_N, z_MM]
            ]
        """
        return np.asarray(self._layers.tolist())

    @property
    def n_cells(self):
        return self._layers.shape[0]

    @property
    def prisms(self) -> np.ndarray:
        """
        :obj:`numpy.array`, shape(*, 5) detailing the assembly of :obj:
        `geoh5py.objects.drape_model.Drapemodel.layers` within the trace
        of the drape model.

        Columns: Easting, Northing, Elevation (top),
        layer index (first), layer count.

        .. code-block:: python

            prisms = [
                [e_1, n_1, z_1, l_1, c_1],
                ...,
                [e_N, n_N, z_N, l_N, c_N]
            ]

        """
        return np.array(self._prisms.tolist())

    @property
    def rotation(self):
        """
        :obj:`numpy.array` of :obj:`float`, shape (3, ): Coordinates of the rotation.
        """
        return None

    @rotation.setter
    def rotation(self, value):
        pass

    @property
    def origin(self):
        """
        :obj:`numpy.array` of :obj:`float`, shape (3, ): Coordinates of the origin.
        """
        return None

    @origin.setter
    def origin(self, value):
        pass

    @classmethod
    def validate_prisms(cls, values) -> np.ndarray:
        """
        Validate and format type of prisms array.

        :param values: Array of prisms as defined by
            :obj:`~geoh5py.objects.drape_model.DrapeModel.prisms`.
        """
        if isinstance(values, (list, tuple)):
            values = np.array(values, ndmin=2)

        if not isinstance(values, np.ndarray):
            raise TypeError(
                "Attribute 'prisms' must be a list, tuple or numpy array. "
                f"Object of type {type(values)} provided."
            )

        if np.issubdtype(values.dtype, np.number):
            if values.shape[1] != 5:
                raise ValueError(
                    "Array of 'prisms' must be of shape (*, 5). "
                    f"Array of shape {values.shape} provided."
                )

            values = np.asarray(
                np.core.records.fromarrays(
                    values.T.tolist(),
                    dtype=cls.__PRISM_DTYPE,
                )
            )

        if values.dtype != cls.__PRISM_DTYPE:
            raise ValueError(
                f"Array of 'prisms' must be of dtype = {cls.__PRISM_DTYPE}"
            )

        return values

    @classmethod
    def validate_layers(cls, values: np.ndarray | list | tuple) -> np.ndarray:
        """
        Validate and format type of layers array.

        :param values: Array of layers as defined by
            :obj:`~geoh5py.objects.drape_model.DrapeModel.layers`.
        """
        if isinstance(values, (list, tuple)):
            values = np.array(values, ndmin=2)

        if not isinstance(values, np.ndarray):
            raise TypeError(
                "Attribute 'layers' must be a list, tuple or numpy array. "
                f"Object of type {type(values)} provided."
            )

        if np.issubdtype(values.dtype, np.number):
            if values.shape[1] != 3:
                raise ValueError(
                    "Array of 'layers' must be of shape (*, 3). "
                    f"Array of shape {values.shape} provided."
                )

            if any(np.diff(np.unique(values[:, 0])) != 1):
                msg = "Prism index (first column) must be monotonically increasing."
                raise ValueError(msg)

            values = np.asarray(
                np.core.records.fromarrays(
                    values.T.tolist(),
                    dtype=cls.__LAYERS_DTYPE,
                )
            )

        if values.dtype != cls.__LAYERS_DTYPE:
            raise ValueError(
                f"Array of 'layers' must be of dtype = {cls.__LAYERS_DTYPE}"
            )

        return values
