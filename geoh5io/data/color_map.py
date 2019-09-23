from typing import Dict


class RGBAColor:
    """ Color in RGBA color space. Coordinates are floating point values between 0 and 1.

    See ``colorsys`` module to convert between color spaces.
    """

    def __init__(self, red: float, green: float, blue: float, alpha=1):
        self._red = red
        self._green = green
        self._blue = blue
        self._alpha = alpha

    @property
    def red(self) -> float:
        return self._red

    @property
    def green(self) -> float:
        return self._green

    @property
    def blue(self) -> float:
        return self._blue

    @property
    def alpha(self) -> float:
        return self._alpha


class ColorMap:
    """ Records colors assigned to value ranges (where Value is the start of the range).
    """

    def __init__(self, color_map: Dict[float, RGBAColor] = None):
        self._map = dict() if color_map is None else color_map

    def __getitem__(self, item: float) -> RGBAColor:
        return self._map[item]

    def __len__(self):
        return len(self._map)
