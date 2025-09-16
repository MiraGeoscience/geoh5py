# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
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

import uuid

import numpy as np

from .grid_object import GridObject
from .object_base import ObjectBase


class DrapeModel(ObjectBase):
    """
    Drape (curtain) model object made up of layers and prisms.

    :param layers: Array of layers in the drape model organized into blocks
        representing each prism in the model.
    :param prisms: Array detailing the assembly of
        :obj:`geoh5py.objects.drape_model.DrapeModel.layers` within the trace
        of the drape model.
    """

    _TYPE_UID = uuid.UUID("{C94968EA-CF7D-11EB-B8BC-0242AC130003}")
    _LAYERS_DTYPE = np.dtype([("I", "<i4"), ("K", "<i4"), ("Bottom elevation", "<f8")])
    _PRISM_DTYPE = np.dtype(
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
        layers: np.ndarray | list | tuple | None = None,
        prisms: np.ndarray | list | tuple | None = None,
        **kwargs,
    ):
        self._centroids: np.ndarray | None = None
        self._layers: np.ndarray = self.validate_layers(layers)
        self._prisms: np.ndarray = self.validate_prisms(prisms)

        super().__init__(**kwargs)

    @property
    def z_cell_size(self) -> np.ndarray:
        """Compute thickness of every cell in the drape model."""
        grid_z = np.full((self.prisms.shape[0], self._layers["K"].max() + 2), np.nan)
        grid_z[:, 0] = self._prisms["Top elevation"]
        grid_z[self._layers["I"], self._layers["K"] + 1] = self._layers[
            "Bottom elevation"
        ]
        hz = (grid_z[:, :-1] - grid_z[:, 1:]).flatten()

        return hz[~np.isnan(hz)]

    @property
    def centroids(self) -> np.ndarray:
        """
        Cell center locations in world coordinates, shape(*, 3).

        .. code-block:: python

            centroids = [
                [x_1, y_1, z_1],
                ...,
                [x_N, y_N, z_N]
            ]
        """
        if getattr(self, "_centroids", None) is None:
            xy = np.repeat(
                np.c_[self.prisms[:, :2]], self.prisms[:, 4].astype(int), axis=0
            )
            z = self.layers[:, 2] + self.z_cell_size / 2.0
            self._centroids = np.c_[xy, z]

        return self._centroids

    def copy(
        self,
        parent=None,
        *,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        **kwargs,
    ) -> DrapeModel:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.copy`.
        """
        mask = self.validate_mask(mask)

        if mask is not None:
            layers = self.layers[mask]
            prisms_ids, new_ids, count = np.unique(
                layers[:, 0], return_inverse=True, return_counts=True
            )
            layers[:, 0] = new_ids
            prisms = self.prisms[prisms_ids.astype(int)]
            prisms[:, 3] = np.r_[0, np.cumsum(count[:-1])]
            prisms[:, 4] = count
            kwargs.update({"prisms": prisms, "layers": layers})

        new_entity = super().copy(
            parent=parent,
            copy_children=copy_children,
            clear_cache=clear_cache,
            mask=mask,
            **kwargs,
        )

        return new_entity

    @property
    def extent(self) -> np.ndarray | None:
        """
        Geography bounding box of the object.

        :return: Bounding box defined by the bottom South-West and
            top North-East coordinates,  shape(2, 3).
        """
        return np.c_[self.prisms.min(axis=0)[:3], self.prisms.max(axis=0)[:3]].T

    @property
    def layers(self) -> np.ndarray:
        """
        Layers in the drape model organized into blocks representing each prism in the model:

        X (prism index), K (depth index), elevation (cell bottom))

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
        if self._layers is None and self.on_file:
            self._layers = self.workspace.fetch_array_attribute(self, "layers")

        return np.asarray(self._layers.tolist())

    @property
    def n_cells(self) -> int:
        return self.layers.shape[0]

    @property
    def prisms(self) -> np.ndarray:
        """
        Array detailing the assembly of :obj:
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
        if self._prisms is None and self.on_file:
            self._prisms = self.workspace.fetch_array_attribute(self, "prisms")

        return np.array(self._prisms.tolist())

    @classmethod
    def validate_prisms(cls, values: np.ndarray | list | tuple | None) -> np.ndarray:
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
            if values.shape[1] != len(cls._PRISM_DTYPE):
                raise ValueError(
                    f"Array of 'prisms' must be of shape (*, {len(cls._PRISM_DTYPE)}). "
                    f"Array of shape {values.shape} provided."
                )

            values = np.asarray(
                np.core.records.fromarrays(
                    values.T.tolist(),
                    dtype=cls._PRISM_DTYPE,
                )
            )

        if values.dtype != cls._PRISM_DTYPE:
            raise ValueError(f"Array of 'prisms' must be of dtype = {cls._PRISM_DTYPE}")

        return values

    @classmethod
    def validate_layers(cls, values: np.ndarray | list | tuple | None) -> np.ndarray:
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
            if values.shape[1] != len(cls._LAYERS_DTYPE):
                raise ValueError(
                    f"Array of 'layers' must be of shape (*, {len(cls._LAYERS_DTYPE)}). "
                    f"Array of shape {values.shape} provided."
                )

            if any(np.diff(np.unique(values[:, 0])) != 1):
                msg = "Prism index (first column) must be monotonically increasing."
                raise ValueError(msg)

            values = np.asarray(
                np.core.records.fromarrays(
                    values.T.tolist(),
                    dtype=cls._LAYERS_DTYPE,
                )
            )

        if values.dtype != cls._LAYERS_DTYPE:
            raise ValueError(
                f"Array of 'layers' must be of dtype = {cls._LAYERS_DTYPE}"
            )

        return values


DrapeModel.validate_mask = GridObject.validate_mask  # type: ignore
DrapeModel.validate_cell_mask = GridObject.validate_cell_mask  # type: ignore
