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


class ArraySource(Protocol):  # pylint: disable=too-few-public-methods
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
        source: ArraySource | None,
        uid: UUID | None,
        key: str,
        *,
        validator: list[ArrayValidator],
        value: np.ndarray | None = None,
        serializer: ArrayValidator | None = None,
    ):
        """
        :param source: Object implementing the :class:`ArraySource` protocol,
            used to fetch the array on demand. May be ``None`` if ``value`` is
            provided directly and the array is never expected to be fetched.
        :param uid: UUID of the entity that owns this array. This is passed
            through to ``source.fetch_array`` so the source knows which
            entity's data to retrieve, and is also used in error messages.
        :param key: The geoh5py attribute name identifying which array on the
            entity to fetch (e.g. ``"vertices"``, ``"values"``). Passed
            through to ``source.fetch_array`` alongside ``uid``.
        :param validator: List of callables applied, in order, to any value
            assigned or fetched. Each validator receives the array returned
            by the previous one, allowing simple validators to be composed
            (e.g. a type/shape coercion validator followed by a stricter
            shape-validation added later via :meth:`with_validator`).
        :param value: An already-loaded array. If provided, it is run through
            ``validator`` immediately and no fetch from ``source`` occurs
            until the value is cleared.
        :param serializer: Optional callable used by :meth:`to_geoh5` to
            convert the in-memory value to the representation expected on
            disk (e.g. a structured dtype). If ``None``, :meth:`to_geoh5`
            returns the value unchanged.
        """
        self.source = source
        self.uid = uid
        self.key = key
        self.validator: list[ArrayValidator] = list(validator)
        self.serializer = serializer
        self._value: np.ndarray | None = None
        if value is not None:
            self.value = value

    def with_validator(self, validator: ArrayValidator) -> LazyArray:
        """
        Attach an additional validator, run after any existing ones, used
        whenever this array is loaded or (re)assigned.
        """
        if validator not in self.validator:
            self.validator.append(validator)
            if self._value is not None:
                self.value = self._value

        return self

    def with_serializer(self, serializer: ArrayValidator) -> LazyArray:
        """
        Attach or replace the serializer used by :meth:`to_geoh5`.
        """
        self.serializer = serializer
        return self

    @property
    def is_loaded(self) -> bool:
        """
        Whether the array has been fetched into memory.
        """
        return self._value is not None

    @property
    def value(self) -> np.ndarray:
        """
        Return the loaded array, fetching it from the source if necessary.
        """
        if self._value is None:
            source = self.source
            uid = self.uid
            if source is None or uid is None:
                raise ValueError(
                    f"Array '{self.key}' for entity '{uid}' has no value "
                    "and no source/uid to load it from."
                )

            fetched = source.fetch_array(uid, self.key)
            if fetched is None:
                raise ValueError(
                    f"Array '{self.key}' for entity '{uid}' could not be loaded."
                )

            self.value = fetched

        value = self._value
        assert value is not None
        return value

    @value.setter
    def value(self, value: np.ndarray) -> None:
        for validator in self.validator:
            value = validator(value)

        self._value = value

    def to_geoh5(self) -> np.ndarray:
        """
        Return the array converted to its on-disk geoh5 representation.

        Applies the ``serializer`` supplied at construction (if any), so a
        model can delegate datatype conversions (e.g. structured arrays) to the
        ``LazyArray`` itself instead of handling them separately.
        """
        if self.serializer is None:
            return self.value

        return self.serializer(self.value)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def dtype(self) -> np.dtype:
        return self.value.dtype

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.value, dtype=dtype)

    def __getitem__(self, item):
        return self.value[item]

    def __len__(self) -> int:
        return len(self.value)

    def __repr__(self) -> str:
        state = "loaded" if self.is_loaded else "lazy"
        return f"LazyArray(key={self.key!r}, uid={self.uid}, state={state})"
