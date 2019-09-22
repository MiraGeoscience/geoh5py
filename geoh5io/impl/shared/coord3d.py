from typing import Tuple


class Coord3D:
    def __init__(self, xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self._xyz = tuple(float(v) for v in xyz[0:3])

    @property
    def x(self) -> float:
        return self._xyz[0]

    @property
    def y(self) -> float:
        return self._xyz[1]

    @property
    def z(self) -> float:
        return self._xyz[2]

    def __getitem__(self, item) -> float:
        return self._xyz[item]
