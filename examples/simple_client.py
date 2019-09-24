#!/usr/bin/env python3
import os
import time

import toml
from thriftpy2.protocol import TBinaryProtocolFactory
from thriftpy2.protocol import TMultiplexedProtocolFactory
from thriftpy2.rpc import client_context

from geoh5io import interfaces


# TODO: share this code between app and client demo
# pylint: disable=too-many-locals
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

    test_size_factor = 1 * int(1e6)
    # list<Coord3D> v/s 3 x list<double>
    # x 1e4 takes 2  / 0
    # x 1e5 takes 17 / 7
    # x 1e6 takes ~150 / 88

    use_coord3d = True
    if use_coord3d:
        base_coords = [
            interfaces.shared.Coord3D(0, 1, 2),
            interfaces.shared.Coord3D(10, 11, 12),
            interfaces.shared.Coord3D(20, 21, 22),
            interfaces.shared.Coord3D(30, 31, 32),
        ]
        coord_list = base_coords * test_size_factor
        assert len(coord_list) == len(base_coords) * test_size_factor

        start = time.time()
        modified_coord_list = objects_service.test_roundtrip(coord_list)
        end = time.time()
        assert len(modified_coord_list) == len(base_coords) * test_size_factor
        print("time elapsed: %d" % (end - start))

        # print('modified_coords:')
        # print(modified_coord_list)
        # print('original_coords:')
        # print(coord_list)

    else:
        coord_list3 = interfaces.objects.CoordList3(
            [0, 10, 20, 30] * test_size_factor,
            [1, 11, 21, 31] * test_size_factor,
            [2, 12, 22, 32] * test_size_factor,
        )
        assert len(coord_list3.x) == 4 * test_size_factor

        start = time.time()
        modified_coord_list3 = objects_service.test_roundtrip2(coord_list3)
        end = time.time()
        assert len(modified_coord_list3.x) == 4 * test_size_factor
        print("time elapsed: %d" % (end - start))

        # print('modified_coords:')
        # print(modified_coord_list3)
        # print('original_coords:')
        # print(coord_list3)

    workspace_service.close()


def main():
    config_path = os.path.join(os.getcwd(), "config.toml")
    config = dict()
    if os.path.exists(config_path):
        config = toml.load(config_path)
        print(f"Configuration loaded from {config_path}")

    port = config.get("PORT", 9090)
    host = config.get("HOST", "localhost")
    timeout_seconds = config.get("TIMEOUT", 30)

    print("My API version: " + interfaces.api.API_VERSION)

    binary_factory = TBinaryProtocolFactory()
    workspace_serv_factory = TMultiplexedProtocolFactory(
        binary_factory, "workspace_thrift"
    )
    objects_serv_factory = TMultiplexedProtocolFactory(binary_factory, "objects_thrift")
    groups_serv_factory = TMultiplexedProtocolFactory(binary_factory, "groups_thrift")
    data_serv_factory = TMultiplexedProtocolFactory(binary_factory, "data_thrift")

    print(f"Starting client on {host}:{port}...")
    with client_context(
        interfaces.workspace.WorkspaceService,
        host,
        port,
        connect_timeout=1000 * timeout_seconds,
        socket_timeout=1000 * timeout_seconds,
        proto_factory=workspace_serv_factory,
    ) as workspace_service:
        with client_context(
            interfaces.objects.ObjectsService,
            host,
            port,
            connect_timeout=1000 * timeout_seconds,
            socket_timeout=1000 * timeout_seconds,
            proto_factory=objects_serv_factory,
        ) as objects_service:
            with client_context(
                interfaces.groups.GroupsService,
                host,
                port,
                connect_timeout=1000 * timeout_seconds,
                socket_timeout=1000 * timeout_seconds,
                proto_factory=groups_serv_factory,
            ) as groups_service:
                with client_context(
                    interfaces.data.DataService,
                    host,
                    port,
                    connect_timeout=1000 * timeout_seconds,
                    socket_timeout=1000 * timeout_seconds,
                    proto_factory=data_serv_factory,
                ) as data_service:
                    simple_demo(
                        workspace_service, objects_service, groups_service, data_service
                    )


if __name__ == "__main__":
    main()
