#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.

# flake8: noqa

import inspect

from geoh5py.workspace import Workspace

from . import groups, objects
from .groups import CustomGroup

# Define the variable '__version__' at the runtime
try:
    from setuptools_scm import get_version

    # This will fail with LookupError if the package is not installed in
    # editable mode or if Git is not installed.
    __version__ = get_version(root="..", relative_to=__file__)
except (ImportError, LookupError):
    # As a fallback, use the version that is hard-coded in the file.
    try:
        from geoh5py._version import __version__  # noqa: F401
    except ModuleNotFoundError:
        # The user is probably trying to run this without having installed
        # the package, so complain.
        raise RuntimeError(
            "Hatch VCS Footgun Example is not correctly installed. "
            "Please install it with pip."
        )

def get_type_uid_classes():
    members = []
    for _, member in inspect.getmembers(groups) + inspect.getmembers(objects):
        if inspect.isclass(member) and hasattr(member, "default_type_uid"):
            members.append(member)

    return members


TYPE_UID_TO_CLASS = {k.default_type_uid(): k for k in get_type_uid_classes()}
