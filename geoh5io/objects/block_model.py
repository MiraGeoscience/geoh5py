import uuid
from typing import Optional

import numpy as np

from .object_base import ObjectBase, ObjectType


class BlockModel(ObjectBase):
    """
    Rectilinear ``BlockModel``, or 3D tensor mesh, is a container for models defined
    by three perpendicular axes. Each axis is divided into discrete
    intervals that define the cell dimensions.
    """

    __TYPE_UID = uuid.UUID(
        fields=(0xB020A277, 0x90E2, 0x4CD7, 0x84, 0xD6, 0x612EE3F25051)
    )
    _attribute_map = ObjectBase._attribute_map.copy()
    _attribute_map.update({"Origin": "origin", "Rotation": "rotation"})

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)
        self._origin = [0, 0, 0]
        self._rotation = 0
        self._u_cell_delimiters = None
        self._v_cell_delimiters = None
        self._z_cell_delimiters = None
        self._centroids = None

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

    @property
    def origin(self) -> np.array:
        """
        Coordinates of the origin: array of floats, shape (3,)
        """
        return self._origin

    @origin.setter
    def origin(self, value):
        if value is not None:
            if isinstance(value, np.ndarray):
                value = value.tolist()

            assert len(value) == 3, "Origin must be a list or numpy array of shape (3,)"

            self.update_h5 = "attributes"
            self._centroids = None

            value = np.asarray(
                tuple(value), dtype=[("x", float), ("y", float), ("z", float)]
            )
            self._origin = value

    @property
    def u_cell_delimiters(self) -> Optional[np.ndarray]:
        """
        Nodal offset along the u-axis: array of floats, shape (u_count,)
        """
        if (
            getattr(self, "_u_cell_delimiters", None) is None
        ) and self.existing_h5_entity:
            delimiters = self.workspace.fetch_delimiters(self.uid)
            self._u_cell_delimiters = delimiters[0]
            self._v_cell_delimiters = delimiters[1]
            self._z_cell_delimiters = delimiters[2]

        return self._u_cell_delimiters

    @u_cell_delimiters.setter
    def u_cell_delimiters(self, value):
        if value is not None:
            value = np.r_[value]
            self.update_h5 = "cell_delimiters"
            self._centroids = None

            self._u_cell_delimiters = value.astype(float)

    @property
    def v_cell_delimiters(self) -> Optional[np.ndarray]:
        """
        Nodal offset along the v-axis: array of floats, shape (u_count,)
        """
        if (
            getattr(self, "_v_cell_delimiters", None) is None
        ) and self.existing_h5_entity:
            delimiters = self.workspace.fetch_delimiters(self.uid)
            self._u_cell_delimiters = delimiters[0]
            self._v_cell_delimiters = delimiters[1]
            self._z_cell_delimiters = delimiters[2]

        return self._v_cell_delimiters

    @v_cell_delimiters.setter
    def v_cell_delimiters(self, value):
        if value is not None:
            value = np.r_[value]
            self.update_h5 = "cell_delimiters"
            self._centroids = None

            self._v_cell_delimiters = value.astype(float)

    @property
    def z_cell_delimiters(self) -> Optional[np.ndarray]:
        """
        Nodal offset along the z-axis: array of floats, shape (u_count,)
        """
        if (
            getattr(self, "_z_cell_delimiters", None) is None
        ) and self.existing_h5_entity:
            delimiters = self.workspace.fetch_delimiters(self.uid)
            self._u_cell_delimiters = delimiters[0]
            self._v_cell_delimiters = delimiters[1]
            self._z_cell_delimiters = delimiters[2]

        return self._z_cell_delimiters

    @z_cell_delimiters.setter
    def z_cell_delimiters(self, value):
        if value is not None:
            value = np.r_[value]
            self.update_h5 = "cell_delimiters"
            self._centroids = None

            self._z_cell_delimiters = value.astype(float)

    @property
    def rotation(self) -> Optional[float]:
        """
        Clockwise rotation angle about the vertical axis: float
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "Rotation angle must be a float of shape (1,)"
            self.update_h5 = "attributes"
            self._centroids = None

            self._rotation = value.astype(float)

    @property
    def u_cells(self) -> Optional[np.ndarray]:
        """
        Cell size along the u-axis: array, shape (u_count,)
        """
        if self.u_cell_delimiters is not None:
            return self.u_cell_delimiters[1:] - self.u_cell_delimiters[:-1]
        return None

    @property
    def v_cells(self) -> Optional[np.ndarray]:
        """
        Cell size along the v-axis: array, shape (v_count,)
        """
        if self.v_cell_delimiters is not None:
            return self.v_cell_delimiters[1:] - self.v_cell_delimiters[:-1]
        return None

    @property
    def z_cells(self) -> Optional[np.ndarray]:
        """
        Cell size along the z-axis: array, shape (z_count,)
        """
        if self.z_cell_delimiters is not None:
            return self.z_cell_delimiters[1:] - self.z_cell_delimiters[:-1]
        return None

    @property
    def dimensions(self) -> Optional[list]:
        """
        Number of cells along the u, v and z-axis: list[int], length (3,)
        """

        if (
            self.u_count is not None
            and self.v_count is not None
            and self.z_count is not None
        ):
            return [self.u_count, self.v_count, self.z_count]
        return None

    @property
    def n_cells(self) -> Optional[int]:
        """
        Total number of cells in the model, int
        """

        if self.dimensions:
            return int(np.prod(self.dimensions))
        return None

    @property
    def cell_center_u(self):
        """
        Cell center locations along u-axis: array of floats, shape(u_count,)
        """
        return np.cumsum(self.u_cells) - self.u_cells / 2.0

    @property
    def cell_center_v(self):
        """
        Cell center locations along v-axis: array of floats, shape(v_count,)
        """
        return np.cumsum(self.v_cells) - self.v_cells / 2.0

    @property
    def cell_center_z(self):
        """
        Cell center locations along z-axis: array of floats, shape(z_count,)
        """
        return np.cumsum(self.z_cells) - self.z_cells / 2.0

    @property
    def centroids(self):
        """
        Cell center locations in world coordinates [x_i, y_i, z_i]:
        array of floats, shape(n_cells, 3)
        """

        if getattr(self, "_centroids", None) is None:

            angle = np.deg2rad(self.rotation)
            rot = np.r_[
                np.c_[np.cos(angle), -np.sin(angle), 0],
                np.c_[np.sin(angle), np.cos(angle), 0],
                np.c_[0, 0, 1],
            ]

            u_grid, v_grid, z_grid = np.meshgrid(
                self.cell_center_u, self.cell_center_v, self.cell_center_z
            )

            xyz = np.c_[np.ravel(u_grid), np.ravel(v_grid), np.ravel(z_grid)]

            self._centroids = np.dot(rot, xyz.T).T

            for ind, axis in enumerate(["x", "y", "z"]):
                self._centroids[:, ind] += self.origin[axis]

        return self._centroids

    @property
    def u_count(self) -> Optional[int]:
        """
        Number of cells along u-axis: int
        """
        if self.u_cell_delimiters is not None:
            return int(self.u_cell_delimiters.shape[0] - 1)
        return None

    @property
    def v_count(self) -> Optional[int]:
        """
        Number of cells along v-axis: int
        """
        if self.v_cell_delimiters is not None:
            return int(self.v_cell_delimiters.shape[0] - 1)
        return None

    @property
    def z_count(self) -> Optional[int]:
        """
        Number of cells along z-axis: int
        """
        if self.z_cell_delimiters is not None:
            return int(self.z_cell_delimiters.shape[0] - 1)
        return None
