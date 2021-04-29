import os
import sys


def convert(file_name):
    name = os.path.splitext(file_name)[0]
    os.system(f"pandoc -s {name}.textile -f textile -t html -o {name}.html")
    os.system(f"pandoc -s -t rst --toc {name}.html -o {name}.rst")
    os.remove(f"{name}.html")


if __name__ == "__main__":
    input_file = sys.argv[1]
    convert(input_file)
