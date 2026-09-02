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

"""Pydantic workspace model and geoh5 file-handle lifecycle."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, Self, cast

import h5py
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from .entity import PydanticEntity
from .project import ProjectAttributes
from .root import Root
from .serialization import Geoh5Writer


DEFAULT_PAGE_SIZE = 65_536  # Same as the legacy Workspace default.


class Workspace(BaseModel, AbstractContextManager["Workspace"]):
    """Own project models and manage access to one geoh5 file."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    path: Path | None = None
    name: str = "GEOSCIENCE"
    page_size: int = Field(default=DEFAULT_PAGE_SIZE, strict=True)
    project: ProjectAttributes = Field(default_factory=ProjectAttributes)
    root: Root = Field(default_factory=Root)

    # The live writer owns h5py's unpicklable file handle. Keeping it private
    # separates transient I/O state from the serializable Workspace model.
    _writer: Geoh5Writer | None = PrivateAttr(default=None)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Require one top-level HDF5 group rather than a nested path."""
        if not value or "/" in value:
            raise ValueError("Project name must be non-empty and cannot contain '/'.")
        return value

    @field_validator("page_size")
    @classmethod
    def validate_page_size(cls, value: int) -> int:
        """Mirror the legacy Workspace page-size requirements."""
        if value < 512 or value % 2 != 0:
            raise ValueError("Page size must be a multiple of 2 and at least 512.")
        return value

    @classmethod
    def create(cls, path: str | Path, **kwargs: Any) -> Self:
        """Create and open a new geoh5 file without using legacy Workspace."""
        workspace = cls(path=path, **kwargs)
        return workspace._create()

    def _create(self) -> Self:
        """Open a new HDF5 file and initialize its geoh5 project structure."""
        if self.path is None:
            raise ValueError("A path is required to create a geoh5 file.")

        if self.is_open:
            raise RuntimeError("The Workspace is already open.")

        h5file: h5py.File | None = None
        created = False
        try:
            # Mode "x" preserves creation-only behavior: existing files are
            # rejected instead of being overwritten.
            h5file = h5py.File(
                self.path,
                "x",
                **self.h5_creation_options(),
            )
            created = True
            self._writer = Geoh5Writer.initialize_project(
                h5file,
                name=self.name,
                attributes=self._project_value().h5_attributes(),
                root=self.root,
            )
        except Exception:
            if h5file is not None:
                h5file.close()
            self._writer = None
            if created:
                self.path.unlink(missing_ok=True)
            raise

        return self

    def _project_value(self) -> ProjectAttributes:
        """Return the validated field value despite pylint's FieldInfo inference."""
        return cast(ProjectAttributes, self.project)

    def open(self, mode: str = "r+") -> Self:
        """Open an existing geoh5 file and return this Workspace."""
        if self.is_open:
            return self

        if self.path is None:
            raise ValueError("A path is required to open a geoh5 file.")

        if mode not in {"r", "r+"}:
            raise ValueError("Workspace mode must be 'r' or 'r+'.")

        h5file = h5py.File(self.path, mode)
        try:
            self._writer = Geoh5Writer(h5file)
        except Exception:
            h5file.close()
            raise

        return self

    def close(self) -> None:
        """Close the live HDF5 handle while retaining the Workspace models."""
        writer = self._writer
        self._writer = None
        if writer is not None and writer.h5file.id.valid:
            writer.h5file.close()

    @property
    def is_open(self) -> bool:
        """Whether this Workspace currently owns a valid HDF5 handle."""
        return self._writer is not None and bool(self._writer.h5file.id.valid)

    @property
    def writer(self) -> Geoh5Writer:
        """Return the live writer or explain how to make one available."""
        if not self.is_open or self._writer is None:
            raise RuntimeError("Workspace is closed; call open() before writing.")
        return self._writer

    @property
    def h5file(self) -> h5py.File:
        """Expose the live handle for low-level inspection when needed."""
        return self.writer.h5file

    def write(
        self,
        model: PydanticEntity,
        *,
        compression: int = 5,
    ) -> h5py.Group:
        """Write an entity through the Workspace's live generic writer."""
        return self.writer.write(model, compression=compression)

    def h5_creation_options(self) -> dict[str, Any]:
        """Return the HDF5 options used by the legacy Workspace creator."""
        return {
            "fs_strategy": "page",
            "page_buf_size": self.page_size * 256,
            "fs_page_size": self.page_size,
            "libver": ("v110", "v114"),
        }

    def __enter__(self) -> Self:
        if not self.is_open:
            self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        """Exclude the live writer so an open Workspace remains picklable."""
        state = super().__getstate__()
        private = dict(state.get("__pydantic_private__") or {})
        private["_writer"] = None
        state["__pydantic_private__"] = private
        return state
