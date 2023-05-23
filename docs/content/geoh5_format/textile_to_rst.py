import os
import sys
from pathlib import Path


def convert(file_name):
    name = Path(file_name).stem
    os.system(f"pandoc -s {name}.textile -f textile -t html -o {name}.html")
    os.system(f"pandoc -s -t rst --toc {name}.html -o {name}.rst")
    os.remove(f"{name}.html")


if __name__ == "__main__":
    input_file = sys.argv[1]
    convert(input_file)
