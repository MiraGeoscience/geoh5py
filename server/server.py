#!/usr/bin/env python3

from geoh5io import interfaces, workspace_handler
from thriftpy2.rpc import make_server


def main():
    handler = workspace_handler.WorkspaceHandler()

    timeout_seconds = 5 * 60
    # TODO: get server address, port and timeout from environment variables
    server = make_server(
        interfaces.workspace.WorkspaceService,
        handler,
        "localhost",
        9090,
        client_timeout=1000 * timeout_seconds,
    )

    # TODO: start a multiplex server for multiple services

    print("Starting the server...")
    server.serve()
    print("done.")


if __name__ == "__main__":
    main()
