import uuid
from typing import Optional

import numpy as np

from geoh5io.shared import Coord3D

from .object_base import ObjectBase, ObjectType


class Octree(ObjectBase):
    __TYPE_UID = uuid.UUID(
        fields=(0x4EA87376, 0x3ECE, 0x438B, 0xBF, 0x12, 0x3479733DED46)
    )

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)
        self._origin = Coord3D()
        self._rotation = 0
        self._u_count = None
        self._v_count = None
        self._w_count = None
        self._u_cell_size = None
        self._v_cell_size = None
        self._w_cell_size = None
        self._octree_cells = None
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
                self.update_h5 = True
            self._centroids = None

            if isinstance(value, list):
                value = np.asarray(
                    tuple(value), dtype=[("x", float), ("y", float), ("z", float)]
                )
            self._origin = value

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
                self.update_h5 = True
            self._centroids = None

            self._rotation = value.astype(float)

    @property
    def u_count(self) -> Optional[int]:
        """
        u_count

        Returns
        -------
        u_count: int
            Number of base cells along u-axis
        """
        return self._u_count

    @u_count.setter
    def u_count(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "u_count must be type(int) of shape (1,)"

            if self.existing_h5_entity:
                self.update_h5 = True
            self._centroids = None

            self._u_count = int(value)

    @property
    def v_count(self) -> Optional[int]:
        """
        v_count

        Returns
        -------
        v_count: int
            Number of base cells along v-axis
        """
        return self._v_count

    @v_count.setter
    def v_count(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "v_count must be type(int) of shape (1,)"

            if self.existing_h5_entity:
                self.update_h5 = True
            self._centroids = None

            self._v_count = int(value)

    @property
    def w_count(self) -> Optional[int]:
        """
        w_count

        Returns
        -------
        w_count: int
            Number of base cells along w-axis
        """
        return self._w_count

    @w_count.setter
    def w_count(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "w_count must be type(int) of shape (1,)"

            if self.existing_h5_entity:
                self.update_h5 = True
            self._centroids = None

            self._w_count = int(value)

    @property
    def u_cell_size(self) -> Optional[float]:
        """
        u_cell_size

        Returns
        -------
        u_cell_size: float
            Cell size along the u-coordinate
        """
        return self._u_cell_size

    @u_cell_size.setter
    def u_cell_size(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "u_cell_size must be type(float) of shape (1,)"

            if self.existing_h5_entity:
                self.update_h5 = True
            self._centroids = None

            self._u_cell_size = value.astype(float)

    @property
    def v_cell_size(self) -> Optional[float]:
        """
        v_cell_size

        Returns
        -------
        v_cell_size: float
            Cell size along the v-coordinate
        """
        return self._v_cell_size

    @v_cell_size.setter
    def v_cell_size(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "v_cell_size must be type(float) of shape (1,)"

            if self.existing_h5_entity:
                self.update_h5 = True
            self._centroids = None

            self._v_cell_size = value.astype(float)

    @property
    def w_cell_size(self) -> Optional[float]:
        """
        w_cell_size

        Returns
        -------
        w_cell_size: float
            Cell size along the w-coordinate
        """
        return self._w_cell_size

    @w_cell_size.setter
    def w_cell_size(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "w_cell_size must be type(float) of shape (1,)"

            if self.existing_h5_entity:
                self.update_h5 = True
            self._centroids = None

            self._w_cell_size = value.astype(float)

    @property
    def octree_cells(self) -> Optional[np.ndarray]:
        """
        octree_cells

        Returns
        -------
        octree_cells: numpy.ndarray(int) of shape (nC, 4)
            Array defining the i,j,k ordering and cell dimensions
            [i, j, k, n_cells]
        """
        if (getattr(self, "_octree_cells", None) is None) and self.existing_h5_entity:
            octree_cells = self.workspace.fetch_octree_cells(self.uid)
            self._octree_cells = octree_cells

        return self._octree_cells

    @octree_cells.setter
    def octree_cells(self, value):
        if value is not None:
            value = np.r_[value]

            if self.existing_h5_entity:
                self.update_h5 = True
            self._centroids = None

            self._octree_cells = value.astype(
                [("I", "<i4"), ("J", "<i4"), ("K", "<i4"), ("NCells", "<i4")]
            )

    @property
    def dimensions(self) -> Optional[list]:
        """
        dimension

        Returns
        -------
        dimension: int
            Number of cells along the u, v and w-axis
        """

        if (
            self.u_count is not None
            and self.v_count is not None
            and self.w_count is not None
        ):
            return [self.u_count, self.v_count, self.w_count]
        return None

    @property
    def centroids(self):
        """
        centroids
        Cell center locations of each cell

        Returns
        -------
        centroids: array of floats, shape(nC, 3)
            The cell center locations [x_i, y_i, z_i]

        """

        if getattr(self, "_centroids", None) is None:

            assert self.octree_cells is not None, "octree_cells must be set"
            assert self.u_cell_size is not None, "u_cell_size must be set"
            assert self.v_cell_size is not None, "v_cell_size must be set"
            assert self.w_cell_size is not None, "w_cell_size must be set"

            angle = np.deg2rad(self.rotation)
            rot = np.r_[
                np.c_[np.cos(angle), -np.sin(angle), 0],
                np.c_[np.sin(angle), np.cos(angle), 0],
                np.c_[0, 0, 1],
            ]

            u_grid = (
                self.octree_cells["I"] + self.octree_cells["NCells"] / 2.0
            ) * self.u_cell_size
            v_grid = (
                self.octree_cells["J"] + self.octree_cells["NCells"] / 2.0
            ) * self.v_cell_size
            w_grid = (
                self.octree_cells["K"] + self.octree_cells["NCells"] / 2.0
            ) * self.w_cell_size

            xyz = np.c_[u_grid, v_grid, w_grid]

            self._centroids = np.dot(rot, xyz.T).T

            for ind, axis in enumerate(["x", "y", "z"]):
                self._centroids[:, ind] += self.origin[axis]

        return self._centroids

    @property
    def n_cells(self) -> Optional[int]:
        """
        n_cells

        Returns
        -------
            n_cells: int
                Number of cells
        """

        if self.octree_cells is not None:
            return self.octree_cells.shape[0]
        return None

    def refine_cells(self, indices):

        octree_cells = self.octree_cells.copy()

        mask = np.ones(self.n_cells, dtype=bool)
        mask[indices] = 0

        new_cells = np.array([], dtype=self.octree_cells.dtype)

        copy_val = []
        for ind in indices:

            level = int(octree_cells[ind][3] / 2)

            if level < 1:
                continue

            # Brake into 8 cells
            for k in range(2):
                for j in range(2):
                    for i in range(2):

                        new_cell = np.array(
                            (
                                octree_cells[ind][0] + i * level,
                                octree_cells[ind][1] + j * level,
                                octree_cells[ind][2] + k * level,
                                level,
                            ),
                            dtype=octree_cells.dtype,
                        )
                        new_cells = np.hstack([new_cells, new_cell])

            copy_val.append(np.ones(8) * ind)

        ind_data = np.hstack(
            [np.arange(self.n_cells)[mask], np.hstack(copy_val)]
        ).astype(int)
        self._octree_cells = np.hstack([octree_cells[mask], new_cells])
        self.entity_type.workspace.sort_children_data(self, ind_data)
