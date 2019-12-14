import uuid

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
        self._rotation = 0
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
    def origin(self, xyz):
        assert len(xyz) == 3, "Origin must be a list"
        self._origin = xyz

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
        assert value.dtype == "float", "u_size must be a float"
        self._u_size = value

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
        assert value.dtype == "float", "v_size must be a float"
        self._v_size = value

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

        assert value.dtype in ["int32", "uint32"], "u_count must be an integer"
        self._u_count = value

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
        assert value.dtype in ["int32", "uint32"], "v_count must be an integer"
        self._v_count = value

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
        assert value.dtype == "float", "Rotation angle must be a float"
        self._rotation = value

    @property
    def is_vertical(self) -> bool:
        return self._is_vertical

    @is_vertical.setter
    def is_vertical(self, value: bool):
        assert isinstance(value, bool), "is_vertical must be of type 'bool'"
        self._is_vertical = value
