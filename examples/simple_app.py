#!/usr/bin/env python3

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

from geoh5py.handlers import (
    DataHandler,
    GroupsHandler,
    ObjectsHandler,
    WorkspaceHandler,
)


# TODO: share this code between app and client demo
def simple_demo(workspace_service, objects_service, groups_service, data_service):
    print("API version: " + workspace_service.get_api_version().value)

    workspace_service.open_geoh5("test.geoh5")
    all_objects = objects_service.get_all()
    print(f"Found {len(all_objects)} Objects in workspace.")

    all_groups = groups_service.get_all()
    print(f"found {len(all_groups)} Groups in workspace.")

    all_data = data_service.get_all()
    print(f"found {len(all_data)} Data in workspace.")

    # TODO: some more interesting examples

    workspace_service.close()


def main():

    workspace_service = WorkspaceHandler()
    objects_service = ObjectsHandler()
    groups_service = GroupsHandler()
    data_service = DataHandler()
    simple_demo(workspace_service, objects_service, groups_service, data_service)


if __name__ == "__main__":
    main()
