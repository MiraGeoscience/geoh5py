#!/usr/bin/env python
# Copyright 2015 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Retrieved from https://github.com/Google/oauth2client
"""Fetch the most recent GAE SDK and decompress it in the current directory.
Usage:
    fetch_gae_sdk.py [<dest_dir>]
Current releases are listed here:
    https://www.googleapis.com/storage/v1/b/appengine-sdks/o?prefix=featured
"""

import json
import sys
import zipfile
from pathlib import Path

import StringIO
import urllib2

_SDK_URL = "https://www.googleapis.com/storage/v1/b/appengine-sdks/o?prefix=featured"


def get_gae_versions():
    try:
        version_info_json = urllib2.urlopen(_SDK_URL).read()
    except:
        return {}
    try:
        version_info = json.loads(version_info_json)
    except:
        return {}
    return version_info.get("items", {})


def _version_tuple(v):
    version_string = Path(v["name"]).stem.rpartition("_")[2]
    return tuple(int(x) for x in version_string.split("."))


def get_sdk_urls(sdk_versions):
    python_releases = [
        v for v in sdk_versions if v["name"].startswith("featured/google_appengine")
    ]
    current_releases = sorted(python_releases, key=_version_tuple, reverse=True)
    return [release["mediaLink"] for release in current_releases]


def main(argv):
    if len(argv) > 2:
        print("Usage: {} [<destination_dir>]".format(argv[0]))
        return 1
    dest_dir = Path(argv[1]) if len(argv) > 1 else Path()
    dest_dir.mkdir(exist_ok=True)

    if (dest_dir / "google_appengine").exists():
        print(f"GAE SDK already installed at {dest_dir}, exiting.")
        return 0

    sdk_versions = get_gae_versions()
    if not sdk_versions:
        print("Error fetching GAE SDK version info")
        return 1
    sdk_urls = get_sdk_urls(sdk_versions)
    for sdk_url in sdk_urls:
        try:
            sdk_contents = StringIO.StringIO(urllib2.urlopen(sdk_url).read())
            break
        except:
            pass
    else:
        print(f"Could not read SDK from any of {sdk_urls}")
        return 1
    sdk_contents.seek(0)
    try:
        zip_contents = zipfile.ZipFile(sdk_contents)
        zip_contents.extractall(dest_dir)
        print(f"GAE SDK Installed to {dest_dir}.")
    except:
        print("Error extracting SDK contents")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[:]))
