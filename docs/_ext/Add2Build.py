import os
import shutil
from pathlib import Path


def copyImages():
    # get relevant directories
    cwd = Path()
    contentdir = cwd / "content"
    buildimagesdir = cwd / "_build" / "html" / "_images"

    # check if images directory exists
    buildimagesdir.mkdir(exist_ok=True)

    # images that have been copied
    imnames = os.listdir(buildimagesdir)

    # search for images that have been missed
    for root, dirList, fileList in os.walk(str(contentdir)):
        if root.endswith("images"):
            for filename in fileList:
                if filename not in imnames:
                    shutil.copy(Path(root) / filename, buildimagesdir)
    return


if __name__ == "__main__":
    copyImages()
