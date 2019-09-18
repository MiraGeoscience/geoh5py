#!/usr/bin/env python3

from geoh5io import interfaces
from thriftpy2.rpc import make_client


def main():

    timeout_seconds = 5 * 60
    # TODO: get server address, port and timeout from environment variables

    client: interfaces.workspace.WorkspaceService = make_client(
        interfaces.workspace.WorkspaceService,
        "localhost",
        9090,
        timeout=1000 * timeout_seconds,
    )

    print("My API version: " + interfaces.api.API_VERSION)
    print("Server API version: " + client.get_api_version().value)

    # TODO: some more interesting examples

    client.close()


if __name__ == "__main__":
    main()
