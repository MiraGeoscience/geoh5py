#!/usr/bin/env python3

import os

import toml

from geoh5io import interfaces, objects_handler, workspace_handler
from thriftpy2.protocol import TBinaryProtocolFactory
from thriftpy2.server import TThreadedServer
from thriftpy2.thrift import TMultiplexedProcessor, TProcessor
from thriftpy2.transport import TBufferedTransportFactory, TServerSocket


def main():
    config_path = os.path.join(os.getcwd(), "config.toml")
    config = dict()
    if os.path.exists(config_path):
        config = toml.load(config_path)
        print(f"Configuration loaded from {config_path}")

    port = config.get("PORT", 9090)
    host = config.get("HOST", "localhost")
    timeout_seconds = config.get("TIMEOUT", 30)

    workspace_proc = TProcessor(
        interfaces.workspace.WorkspaceService, workspace_handler.WorkspaceHandler()
    )
    objects_proc = TProcessor(
        interfaces.objects.ObjectService, objects_handler.ObjectsHandler()
    )

    mux_proc = TMultiplexedProcessor()
    mux_proc.register_processor("workspace_thrift", workspace_proc)
    mux_proc.register_processor("objects_thrift", objects_proc)

    server = TThreadedServer(
        mux_proc,
        TServerSocket(host, port, client_timeout=1000 * timeout_seconds),
        iprot_factory=TBinaryProtocolFactory(),
        itrans_factory=TBufferedTransportFactory(),
    )

    print(f"Starting server on {host}:{port}...")
    server.serve()
    print("done.")


if __name__ == "__main__":
    main()
