# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2026 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

import numpy as np


# Purpose of lazy loading:
# let PointsModel hold vertices without necessarily loading them into memory right away.
# Maybe also allow for accessing/modifying only a specific section of the array.


ArrayValidator = Callable[[np.ndarray], np.ndarray]


class ArraySource(Protocol):  # pylint: disable=R0903
    """
    Minimal interface for fetching large entity arrays on demand.

    This intentionally avoids depending on :class:`geoh5py.workspace.Workspace`.
    A concrete source can fetch from a live Workspace, an HDF5 file, an object
    store, or an in-memory dictionary.

    Protocol means anything with a method shaped like fetch_array counts as an ArraySource.
    Later we could have specific
    classes e.g. Geoh5ArraySource for h5 file reading or WorkspaceArraySource for workspace reading,
    and any of these would work so long as they follow the same protocol.
    """

    def fetch_array(self, uid: UUID, key: str) -> np.ndarray | None:
        """
        Fetch an array by entity uid and geoh5py attribute key.
        """


@dataclass(slots=True)
class CallableArraySource:
    """
    Small adapter for notebook experiments and tests.

    Can pass in a function like this:

    def fetcher(uid, key):
        # logic to fetch array
        return array

    source = CallableArraySource(fetcher)

    then doing source.fetch_array(uid, "vertices") will call fetcher(uid, "vertices")
    and return the result.

    """

    fetcher: Callable[[UUID, str], np.ndarray | None]

    def fetch_array(self, uid: UUID, key: str) -> np.ndarray | None:
        return self.fetcher(uid, key)


class LazyArray:
    """
    Array proxy that loads and validates only when accessed.

    The proxy is small, it provides numpy coercion and a few common
    array conveniences, while keeping ownership of IO outside the pydantic model.
    """

    def __init__(
        self,
        source: ArraySource,
        uid: UUID,
        key: str,
        *,
        validator: ArrayValidator | None = None,
        value: np.ndarray | None = None,
    ):
        self.source = source
        self.uid = uid
        self.key = key
        self.validator = validator
        self._value = self._validate(value) if value is not None else None

    def _validate(self, value: np.ndarray) -> np.ndarray:
        if self.validator is None:
            return value

        return self.validator(value)

    def with_validator(self, validator: ArrayValidator) -> LazyArray:
        """
        Attach or replace the validator used when loading this array.
        """
        self.validator = validator
        if self._value is not None:
            self._value = self._validate(self._value)

        return self

    @property
    def is_loaded(self) -> bool:
        """
        Whether the array has been fetched into memory.
        """
        return self._value is not None

    def load(self) -> np.ndarray:
        """
        Return the loaded array, fetching it from the source if necessary.
        """
        if self._value is None:
            value = self.source.fetch_array(self.uid, self.key)
            if value is None:
                raise ValueError(
                    f"Array '{self.key}' for entity '{self.uid}' could not be loaded."
                )

            self._value = self._validate(value)

        return self._value

    @property
    def shape(self) -> tuple[int, ...]:
        return self.load().shape

    @property
    def dtype(self) -> np.dtype:
        return self.load().dtype

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.load(), dtype=dtype)

    def __getitem__(self, item):
        return self.load()[item]

    def __len__(self) -> int:
        return len(self.load())

    def __repr__(self) -> str:
        state = "loaded" if self.is_loaded else "lazy"
        return f"LazyArray(key={self.key!r}, uid={self.uid}, state={state})"
