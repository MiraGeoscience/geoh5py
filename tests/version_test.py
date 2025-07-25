# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
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

from pathlib import Path

import tomli as toml
import yaml
from jinja2 import Template
from packaging.version import InvalidVersion, Version

import geoh5py


def get_conda_recipe_version():
    path = Path(__file__).resolve().parents[1] / "recipe.yaml"
    with open(str(path), encoding="utf-8") as file:
        content = file.read()

    template = Template(content)
    rendered_yaml = template.render()

    recipe = yaml.safe_load(rendered_yaml)

    return recipe["context"]["version"]


def test_version_is_consistent():
    project_version = Version(geoh5py.__version__)
    conda_version = Version(get_conda_recipe_version())
    assert conda_version.base_version == project_version.base_version

    if project_version.base_version != Version("0.0.0").base_version:
        assert conda_version.is_devrelease == project_version.is_devrelease
        assert conda_version.is_prerelease == project_version.is_prerelease
        assert conda_version.is_postrelease == project_version.is_postrelease
        assert conda_version == project_version


def test_conda_version_is_pep440():
    version = Version(get_conda_recipe_version())
    assert version is not None


def validate_version(version_str):
    try:
        version = Version(version_str)
        return (version.major, version.minor, version.micro, version.pre, version.post)
    except InvalidVersion:
        return None


def test_version_is_valid():
    assert validate_version(geoh5py.__version__) is not None
