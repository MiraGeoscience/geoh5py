#!/usr/bin/env python3

#  Copyright (c) 2022 Mira Geoscience Ltd.
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

import re
import sys
from datetime import date

if __name__ == "__main__":
    current_year = date.today().year
    copyright_re = re.compile(rf"\bcopyright \(c\) \b{current_year}\b", re.IGNORECASE)
    files = sys.argv[1:]
    max_lines = 10
    report_files = []
    for f in files:
        with open(f, encoding="utf-8") as file:
            count = 0
            has_dated_copyright = False
            for line in file:
                count += 1
                if count >= max_lines:
                    break
                if re.search(copyright_re, line):
                    has_dated_copyright = True
                    break

            if not has_dated_copyright:
                report_files.append(f)

    if len(report_files) > 0:
        for f in report_files:
            sys.stderr.write(f"{f}: No copyright or invalid year\n")
        exit(1)

# readonly CURRENT_YEAR=$(date +"%Y")

# if ! grep -e "Copyright (c) .*$CURRENT_YEAR" $(head -10 $f) 2>&1 1>/dev/null; then
#    echo "File '$f' has no copyright or an invalid year"
#    exit 1
# fi
