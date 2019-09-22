import uuid

import numpy as np


class H5Reader:
    @staticmethod
    def bool_value(value: np.uint8) -> bool:
        return bool(value)

    @staticmethod
    def uuid_value(value: str) -> uuid.UUID:
        return uuid.UUID(value)
