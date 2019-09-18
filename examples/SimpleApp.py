#!/usr/bin/env python3

from geoh5io import workspace_handler


def main():

    workspace_handler = workspace_handler.WorkspaceHandler()
    print("API version: " + workspace_handler.get_api_version().value)

    # TODO: some more interesting examples


if __name__ == "__main__":
    main()
