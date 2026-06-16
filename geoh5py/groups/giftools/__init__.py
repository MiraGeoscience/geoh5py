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
"""
GIFtools application groups.

How to add support for a new GIFtools group:
-----

1. Create a GIFtools group in Geoscience ANALYST (the outer GIFtools group, and the inner group you
    would like to add support for within that)

2. Copy its parameters using (from the actual "GIFtools" group, not the inner one) into
   a module-level ``<NAME>_PARAMETERS`` dict.  For example, if you want to add support for the group
   named "maginv3d_60", copy from the parameters of the object with the named "GIFtools" and not the
   object with the name "maginv3d_60".

   Keep the form exactly as ANALYST writes it. Object selectors such as ``mesh``
   rely on the rich form (notably ``meshType``) to bind and auto-select the stored entity.

3. Reset instance-specific fields (``uuid``, ``working_directory`` and the object
   ``value`` entries) to empty.

4. Subclass :obj:`~geoh5py.groups.giftools.base.BaseGIFtoolsGroup` and set
   ``_TYPE_UID``, ``_default_name`` and ``_default_parameters``.

5. Export the class from :mod:`geoh5py.groups` so it is registered for
   read/write.

6. Add the class to ``tests/groups/giftools_test.py`` with the mesh type it
   expects (octree groups -> :obj:`~geoh5py.objects.Octree`; tensor-mesh groups
   -> :obj:`~geoh5py.objects.BlockModel`, etc.).
"""
