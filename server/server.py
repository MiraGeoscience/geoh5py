#!/usr/bin/env python3

import os

import toml

from geoh5io import interfaces, workspace_handler
from thriftpy2.rpc import make_server


def main():
    config_path = os.path.join(os.getcwd(), "config.toml")
    config = dict()
    if os.path.exists(config_path):
        config = toml.load(config_path)
        print(f"Configuration loaded from {config_path}")

    port = config.get("PORT", 9090)
    host = config.get("HOST", "localhost")
    timeout_seconds = config.get("TIMEOUT", 30)

    handler = workspace_handler.WorkspaceHandler()

    # TODO: get server address, port and timeout from environment variables
    server = make_server(
        interfaces.workspace.WorkspaceService,
        handler,
        host,
        port,
        client_timeout=1000 * timeout_seconds,
    )

    # TODO: start a multiplex server for multiple services

    print(f"Starting server on {host}:{port}...")
    server.serve()
    print("done.")


if __name__ == "__main__":
    main()
