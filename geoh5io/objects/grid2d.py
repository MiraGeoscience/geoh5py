import uuid
from typing import Optional

import numpy as np

from .object_base import ObjectBase, ObjectType


class Grid2D(ObjectBase):
    __TYPE_UID = uuid.UUID(
        fields=(0x48F5054A, 0x1C5C, 0x4CA4, 0x90, 0x48, 0x80F36DC60A06)
    )

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)

        self._origin = None
        self._u_size = None
        self._v_size = None
        self._u_count = None
        self._v_count = None
        self._rotation = 0.0
        self._vertical = False
        self._centroids = None

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @property
    def origin(self):
        """
        origin

        Returns
        -------
        origin: ndarray of floats, shape (3,)
            Coordinates of the origin
        """
        return self._origin

    @origin.setter
    def origin(self, value):
        if value is not None:
            assert len(value) == 3, "Origin must be a list or numpy array of shape (3,)"

            if self.existing_h5_entity:
                self.update_h5 = ["origin"]
            self._centroids = None

            if isinstance(value, list):
                value = np.asarray(
                    tuple(value), dtype=[("x", float), ("y", float), ("z", float)]
                )
            self._origin = value

    @property
    def u_size(self) -> Optional[float]:
        """
        u_size

        Returns
        -------
        u_size: float
            Cell size along the u-coordinate
        """
        return self._u_size

    @u_size.setter
    def u_size(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "u_size must be a float of shape (1,)"

            if self.existing_h5_entity:
                self.update_h5 = ["u_size"]
            self._centroids = None

            self._u_size = value.astype(float)

    @property
    def v_size(self) -> Optional[float]:
        """
        v_size

        Returns
        -------
        v_size: float
            Cell size along the v-coordinate
        """
        return self._v_size

    @v_size.setter
    def v_size(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "v_size must be a float of shape (1,)"

            if self.existing_h5_entity:
                self.update_h5 = ["v_size"]
            self._centroids = None

            self._v_size = value.astype(float)

    @property
    def u_count(self) -> Optional[int]:
        """
        u_count

        Returns
        -------
        u_count: int
            Number of cells along the u-coordinate
        """
        return self._u_count

    @u_count.setter
    def u_count(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "u_count must be an integer of shape (1,)"

            if self.existing_h5_entity:
                self.update_h5 = ["u_count"]
            self._centroids = None

            self._u_count = value.astype(int)

    @property
    def v_count(self) -> Optional[int]:
        """
        v_count

        Returns
        -------
        v_count: int
            Number of cells along the v-coordinate
        """
        return self._v_count

    @v_count.setter
    def v_count(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "v_count must be an integer of shape (1,)"

            if self.existing_h5_entity:
                self.update_h5 = ["v_count"]
            self._centroids = None

            self._v_count = value.astype(int)

    @property
    def rotation(self) -> Optional[float]:
        """
        rotation

        Returns
        -------
        rotation: array of floats, shape (3,)
            Clockwise rotation angle about the vertical axis
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "Rotation angle must be a float of shape (1,)"

            if self.existing_h5_entity:
                self.update_h5 = ["rotation"]
            self._centroids = None

            self._rotation = value.astype(float)

    @property
    def vertical(self) -> Optional[bool]:
        return self._vertical

    @vertical.setter
    def vertical(self, value: bool):
        if value is not None:
            assert isinstance(value, bool) or value in [
                0,
                1,
            ], "vertical must be of type 'bool'"

            if self.existing_h5_entity:
                self.update_h5 = ["vertical"]
            self._centroids = None

            self._vertical = value

    @property
    def n_cells(self) -> Optional[int]:
        """
        n_cells

        Returns
        -------
            n_cells: int
                Number of cells
        """

        assert (self.u_count is not None) and (
            self.v_count is not None
        ), "'u_count' and 'v_count' must be set"

        return int(self.u_count * self.v_count)

    @property
    def cell_center_u(self):
        """
        cell_center_u

        Returns
        -------
        cell_center_u: array of floats, shape(u_count,)
            The cell center location along u

        """
        return np.cumsum(np.ones(self.u_count) * self.u_size) - self.u_size / 2.0

    @property
    def cell_center_v(self):
        """
        cell_center_v

        Returns
        -------
        cell_center_v: array of floats, shape(v_count,)
            The cell center location along v

        """
        return np.cumsum(np.ones(self.v_count) * self.v_size) - self.v_size / 2.0

    @property
    def centroids(self):
        """
        cell_centers

        Returns
        -------
        cell_centers: array of floats, shape(nC, 3)
            The cell center locations [x_i, y_i, z_i]

        """

        if getattr(self, "_centroids", None) is None:

            angle = np.deg2rad(self.rotation)
            rot = np.r_[
                np.c_[np.cos(angle), -np.sin(angle), 0],
                np.c_[np.sin(angle), np.cos(angle), 0],
                np.c_[0, 0, 1],
            ]

            u_grid, v_grid = np.meshgrid(self.cell_center_u, self.cell_center_v)
            if self.vertical:
                xyz = np.c_[np.ravel(u_grid), np.zeros(self.n_cells), np.ravel(v_grid)]

            else:
                xyz = np.c_[np.ravel(u_grid), np.ravel(v_grid), np.zeros(self.n_cells)]

            self._centroids = np.dot(rot, xyz.T).T

            for ind, axis in enumerate(["x", "y", "z"]):
                self._centroids[:, ind] += self.origin[axis]

        return self._centroids
