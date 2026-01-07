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

from geoh5py.groups import ContainerGroup
from geoh5py.objects import Points
from geoh5py.workspace import Workspace


def test_user_comments(tmp_path):
    h5file_path = tmp_path / r"group_object_comment.geoh5"

    with Workspace.create(h5file_path) as workspace:
        object_base = Points.create(workspace, name="myObject")
        object_comment = "object text comment"
        author = "John Doe"
        object_base.add_comment(object_comment, author=author)

        # Read the comments back in
        ws2 = Workspace(h5file_path)
        object_base = ws2.get_entity("myObject")[0]
        assert object_base.comments.values["Comments"][0]["Author"] == author, (
            "Issue with 'Author of object comments"
        )
        assert object_base.comments.values["Comments"][0]["Text"] == object_comment, (
            "Issue with 'Text' of object comments"
        )

        # Repeat with Group comments
        group = ContainerGroup.create(ws2, name="myGroup")
        group_comment_1 = "group text comment"
        group_comment_2 = "my other comment"

        group.add_comment(group_comment_1)
        group.add_comment(group_comment_2)

        ws3 = Workspace(h5file_path)
        group_in = ws3.get_entity("myGroup")[0]

        assert group_in.comments.values["Comments"][0]["Author"] == ",".join(
            ws3.contributors
        ), "Issue with 'Author of object comments"
        assert group_in.comments.values["Comments"][0]["Text"] == group_comment_1, (
            "Issue with 'Text' of group comments"
        )
        assert group_in.comments.values["Comments"][1]["Text"] == group_comment_2, (
            "Issue with 'Text' of group comments"
        )
