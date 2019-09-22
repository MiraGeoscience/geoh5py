#!/usr/bin/env python3
from geoh5io.handlers import DataHandler
from geoh5io.handlers import GroupsHandler
from geoh5io.handlers import ObjectsHandler
from geoh5io.handlers import WorkspaceHandler


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
