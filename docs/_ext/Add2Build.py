#  Copyright (c) 2021 Mira Geoscience Ltd.
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

import os
import shutil


def copyImages():
    # get relevant directories
    cwd = os.getcwd()
    contentdir = cwd + "/content"
    buildimagesdir = cwd + "/_build/html/_images"

    # check if images directory exists
    if not os.path.isdir(buildimagesdir):
        os.mkdir(buildimagesdir)

    # images that have been copied
    imnames = os.listdir(buildimagesdir)

    # search for images that have been missed
    for root, dirList, fileList in os.walk(contentdir):
        if root.endswith("images"):
            for filename in fileList:
                if filename not in imnames:
                    shutil.copy(os.path.join(root, filename), buildimagesdir)
    return


if __name__ == "__main__":
    copyImages()
