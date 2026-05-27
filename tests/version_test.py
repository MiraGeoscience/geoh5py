# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
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

import importlib
import json
import re
from pathlib import Path

import pytest
import yaml
from packaging.version import Version

import geoh5py


def _get_json_version() -> str:
    version_json_path = Path(__file__).resolve().parents[1] / "_version.json"
    with version_json_path.open(encoding="utf-8") as file:
        version_json = json.load(file)
    return version_json["version"]


def _get_conda_recipe_version_def() -> str:
    recipe_path = Path(__file__).resolve().parents[1] / "recipe.yaml"

    with recipe_path.open(encoding="utf-8") as file:
        recipe = yaml.safe_load(file)
    return recipe["context"]["version"]


def _version_module_exists():
    try:
        importlib.import_module("geoh5py._version")
        return True
    except ModuleNotFoundError:
        return False


def test_conda_recipe_version_loads_json():
    conda_version_def = _get_conda_recipe_version_def()
    regex = (
        r"\$\{\{\s*load_from_file\(\s*['\"](_version\.json)['\"]\s*\)"
        r"\s*\.version\b.*\}\}"
    )
    regex_match = re.match(regex, conda_version_def)
    assert regex_match is not None


@pytest.mark.skipif(
    _version_module_exists(),
    reason="geoh5py._version can be found: package is built",
)
def test_fallback_version_is_zero():
    project_version = Version(geoh5py.__version__)
    fallback_version = Version("0.0.0.dev0")
    assert project_version.base_version == fallback_version.base_version
    assert project_version.pre is None
    assert project_version.post is None
    assert project_version.dev == fallback_version.dev


@pytest.mark.skipif(
    not _version_module_exists(),
    reason="geoh5py._version cannot be found: uses a fallback version",
)
def test_version_json_is_consistent():
    project_version = Version(geoh5py.__version__)
    json_version = Version(_get_json_version())
    assert project_version == json_version