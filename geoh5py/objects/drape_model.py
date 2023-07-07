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

import uuid

import numpy as np

from .grid_object import GridObject
from .object_base import ObjectType


class DrapeModel(GridObject):
    """
    Drape (curtain) model object made up of layers and prisms.
    """

    __TYPE_UID = uuid.UUID("{C94968EA-CF7D-11EB-B8BC-0242AC130003}")

    def __init__(self, object_type: ObjectType, **kwargs):
        self._layers: np.ndarray | None = None
        self._prisms: np.ndarray | None = None

        super().__init__(object_type, **kwargs)

        object_type.workspace._register_object(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @property
    def centroids(self):
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
            if self.layers is None:
                raise AttributeError(
                    "Attribute 'layers' must be defined before accessing 'centroids'."
                )

            if self.prisms is None:
                raise AttributeError(
                    "Attribute 'prisms' must be defined before accessing 'centroids'."
                )

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
    def layers(self) -> np.ndarray | None:
        """
        :obj:`~geoh5py.objects.object_base.ObjectBase.layers`
        """
        if self._layers is None and self.on_file:
            self._layers = self.workspace.fetch_array_attribute(self, "layers")

        if self._layers is not None:
            return np.asarray(self._layers.tolist())

        return None

    @layers.setter
    def layers(self, xyz: np.ndarray):
        if any(np.diff(np.unique(xyz[:,])) != 1):
            msg = "Prism index (first column) must be monotonically increasing."
            raise ValueError(msg)

        if xyz.shape[1] != 3:
            msg = f"Array of layers must be of shape (*, 3). Array of shape {xyz.shape} provided."
            raise ValueError(msg)

        self._layers = np.asarray(
            np.core.records.fromarrays(
                xyz.T.tolist(),
                dtype=[("I", "<i4"), ("K", "<i4"), ("Bottom elevation", "<f8")],
            )
        )
        self.workspace.update_attribute(self, "layers")

    @property
    def n_cells(self):
        if self._prisms is not None:
            return int(self._prisms["Layer count"].sum())
        return None

    @property
    def prisms(self) -> np.ndarray | None:
        """
        :obj:`~geoh5py.objects.object_base.ObjectBase.prisms`
        """
        if self._prisms is None and self.on_file:
            self._prisms = self.workspace.fetch_array_attribute(self, "prisms")

        if self._prisms is not None:
            return np.array(self._prisms.tolist())

        return None

    @prisms.setter
    def prisms(self, xyz: np.ndarray):
        assert (
            xyz.shape[1] == 5
        ), f"Array of prisms must be of shape (*, 5). Array of shape {xyz.shape} provided."
        self._prisms = np.asarray(
            np.core.records.fromarrays(
                xyz.T.tolist(),
                dtype={
                    "names": [
                        "Top easting",
                        "Top northing",
                        "Top elevation",
                        "First layer",
                        "Layer count",
                    ],
                    "formats": ["<f8", "<f8", "<f8", "<i4", "<i4"],
                },
            )
        )
        self.workspace.update_attribute(self, "prisms")
