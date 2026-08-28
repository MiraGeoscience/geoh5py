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

"""Create the core HDF5 structure required by a geoh5 project."""

from __future__ import annotations

from pathlib import Path

import h5py

from .project import Geoh5Project
from .root import RootModel
from .serialization import (
    CHILD_COLLECTIONS,
    TYPE_COLLECTIONS,
    Geoh5EntityPayload,
    Geoh5Writer,
)


def create_geoh5(
    path: str | Path,
    *,
    project: Geoh5Project | None = None,
    root: RootModel | None = None,
) -> Path:
    """
    Create a blank geoh5 file without constructing a legacy Workspace.

    The returned path points to a closed, fully initialized file. Entity data
    can subsequently be added by opening the file with h5py and constructing a
    :class:`Geoh5Writer`.

    :param path: New file path. Existing files are never overwritten.
    :param project: Project metadata and HDF5 creation settings.
    :param root: Optional customized Root entity.
    :return: Path to the newly created file.
    """
    output_path = Path(path)
    project = project or Geoh5Project()
    root = root or RootModel()

    # Build the payload before touching the filesystem so model or placement
    # errors cannot leave an empty file behind.
    root_payload = Geoh5EntityPayload.from_model(root)
    if root_payload.collection != "Groups":
        raise ValueError("The geoh5 Root must belong to the Groups collection.")

    created = False
    try:
        # This mirrors Workspace._create_h5. Mode "x" gives creation-only
        # behavior and raises instead of replacing an existing file.
        with h5py.File(
            output_path,
            "x",
            **project.h5_creation_options(),
        ) as h5file:
            created = True
            _initialize_project(h5file, project, root_payload)
    except Exception:
        # If initialization fails after opening the file, remove only the file
        # created by this call rather than leaving a malformed geoh5 behind.
        if created:
            output_path.unlink(missing_ok=True)
        raise

    return output_path


def _initialize_project(
    h5file: h5py.File,
    project: Geoh5Project,
    root_payload: Geoh5EntityPayload,
) -> None:
    """Write project collections, attributes, and the mandatory Root entity."""
    # Create the single top-level project group, normally /GEOSCIENCE.
    project_group = h5file.create_group(project.name, track_order=True)

    # Create /Data, /Groups, and /Objects to own each category of entity.
    for collection in CHILD_COLLECTIONS:
        project_group.create_group(collection, track_order=True)

    # Create /Types with Data types, Group types, and Object types beneath it.
    type_root = project_group.create_group("Types", track_order=True)
    for collection in TYPE_COLLECTIONS:
        type_root.create_group(collection, track_order=True)

    # Root's actual group lives in /Groups/{root_uid}.
    root_uid = Geoh5Writer.format_uuid(root_payload.uid)
    root_group = project_group["Groups"].create_group(root_uid, track_order=True)

    # Create Root's /Data, /Groups, and /Objects child collections.
    for collection in CHILD_COLLECTIONS["Groups"]:
        root_group.create_group(collection, track_order=True)

    # Expose that same group at /Root using an HDF5 hard link, not a copy.
    project_group["Root"] = root_group

    # Reuse the generic writer's encoding rules for all stored values below.
    writer = Geoh5Writer(h5file)

    # Store Contributors, Distance unit, GA Version, and Version on the project.
    writer._write_attributes(  # pylint: disable=protected-access
        project_group,
        project.h5_attributes(),
    )

    # Root's ID, name, permissions, and visibility are group attributes.
    writer._write_attributes(  # pylint: disable=protected-access
        root_group,
        root_payload.attributes,
    )

    # Write optional Root datasets, such as metadata, when present.
    writer._write_datasets(  # pylint: disable=protected-access
        root_group,
        root_payload.datasets,
        compression=5,
    )

    # Create or retrieve Root's shared type under /Types/Group types.
    type_group, _ = writer._ensure_type(  # pylint: disable=protected-access
        root_payload
    )

    # Link Root to the shared type rather than duplicating the type group.
    root_group["Type"] = type_group
