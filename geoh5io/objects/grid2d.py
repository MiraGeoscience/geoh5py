import uuid

from numpy import r_

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
        self._is_vertical = False

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
        value = r_[value]
        assert len(value) == 3, "Origin must be a list or numpy array of shape (3,)"
        self._origin = value.astype(float)

    @property
    def u_size(self):
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
        value = r_[value]
        assert len(value) == 1, "u_size must be a float of shape (1,)"
        self._u_size = value.astype(float)

    @property
    def v_size(self):
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
        value = r_[value]
        assert len(value) == 1, "v_size must be a float of shape (1,)"
        self._v_size = value.astype(float)

    @property
    def u_count(self):
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
        value = r_[value]
        assert len(value) == 1, "u_count must be an integer of shape (1,)"
        self._u_count = value.astype(int)

    @property
    def v_count(self):
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
        value = r_[value]
        assert len(value) == 1, "v_count must be an integer of shape (1,)"
        self._v_count = value.astype(int)

    @property
    def rotation(self):
        """
        rotation

        Returns
        -------
        rotation: ndarray of floats, shape (3,)
            Rotation angle about the vertical axis
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        value = r_[value]
        assert len(value) == 1, "Rotation angle must be a float of shape (1,)"
        self._rotation = value.astype(float)

    @property
    def is_vertical(self) -> bool:
        return self._is_vertical

    @is_vertical.setter
    def is_vertical(self, value: bool):
        assert isinstance(value, bool), "is_vertical must be of type 'bool'"
        self._is_vertical = value
