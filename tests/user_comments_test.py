#  Copyright (c) 2020 Mira Geoscience Ltd.
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

import tempfile
from pathlib import Path

from geoh5py.groups import ContainerGroup
from geoh5py.objects import Points
from geoh5py.workspace import Workspace


def test_user_comments():

    with tempfile.TemporaryDirectory() as tempdir:
        h5file_path = Path(tempdir) / r"group_object_comment.geoh5"

        # Create a workspace
        workspace = Workspace(h5file_path)

        object_base = Points.create(workspace, name="myObject")
        object_comment = "object text comment"
        author = "John Doe"
        object_base.add_comment(object_comment, author=author)

        workspace.finalize()

        # Read the comments back in
        workspace = Workspace(h5file_path)
        object_base = workspace.get_entity("myObject")[0]
        assert (
            object_base.comments.values[0]["Author"] == author
        ), "Issue with 'Author of object comments"
        assert (
            object_base.comments.values[0]["Text"] == object_comment
        ), "Issue with 'Text' of object comments"

        # Repeat with Group comments
        group = ContainerGroup.create(workspace, name="myGroup")
        group_comment_1 = "group text comment"
        group_comment_2 = "my other comment"
        author = "Jane Doe"
        group.add_comment(group_comment_1, author=author)

        group.add_comment(group_comment_2, author=author)

        workspace.finalize()

        workspace = Workspace(h5file_path)
        group_in = workspace.get_entity("myGroup")[0]

        assert (
            group_in.comments.values[0]["Author"] == author
        ), "Issue with 'Author of object comments"
        assert (
            group_in.comments.values[0]["Text"] == group_comment_1
        ), "Issue with 'Text' of group comments"

        assert (
            group_in.comments.values[1]["Text"] == group_comment_2
        ), "Issue with 'Text' of group comments"
