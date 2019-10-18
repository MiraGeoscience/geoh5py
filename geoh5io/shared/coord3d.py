import numpy as np


class Coord3D:
    def __init__(self, xyz: np.array = np.empty((1, 3))):
        self._xyz = xyz

    @property
    def x(self) -> float:
        return self._xyz[:, 0]

    @property
    def y(self) -> float:
        return self._xyz[:, 1]

    @property
    def z(self) -> float:
        return self._xyz[:, 2]

    def __getitem__(self, item) -> float:
        return self._xyz[item, :]
