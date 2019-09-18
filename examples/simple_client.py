#!/usr/bin/env python3

import os

import toml

from geoh5io import interfaces
from thriftpy2.rpc import make_client


def main():
    config_path = os.path.join(os.getcwd(), "config.toml")
    config = dict()
    if os.path.exists(config_path):
        config = toml.load(config_path)
        print(f"Configuration loaded from {config_path}")

    port = config.get("PORT", 9090)
    host = config.get("HOST", "localhost")
    timeout_seconds = config.get("TIMEOUT", 30)

    client: interfaces.workspace.WorkspaceService = make_client(
        interfaces.workspace.WorkspaceService,
        host,
        port,
        timeout=1000 * timeout_seconds,
    )

    print(f"Starting client on {host}:{port}...")

    print("My API version: " + interfaces.api.API_VERSION)
    print("Server API version: " + client.get_api_version().value)

    # TODO: some more interesting examples

    client.close()
    print("Done.")


if __name__ == "__main__":
    main()
