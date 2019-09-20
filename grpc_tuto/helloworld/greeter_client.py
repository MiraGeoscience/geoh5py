# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function
import logging

import grpc

import helloworld_pb2
import helloworld_pb2_grpc

import time
import numpy

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        response = stub.SayHello(helloworld_pb2.HelloRequest(name='you'))
        print("Greeter client received: " + response.message)

        test_size_factor = int(1e5) # message too large above 3*1e4.  Streaming appear to be slow, but could stream per
        # chunk although the number elements allowed in a chunk depends on the  size in memory of the each element:
        # seems hard to find the right chunk size...
        #
        # CoordList v/s CoordList2 v/s stream client
        # x 1e4 takes 1  / 0 / 13 second
        # x 1e5 takes ?? / ? / 135 seconds
        # x 1e6 takes ??? / ?? / 2 seconds

        base_coords = [[0, 1, 2], [10, 11, 12], [20, 21, 22], [30, 31, 32]]
        if False:
            input_coords = base_coords * test_size_factor
            start = time.time()
            modified_coords = stub.TranslateCoord3List(
                helloworld_pb2.TranslateCoord3ListRequest(
                    coord_list = helloworld_pb2.Coord3List(
                        coords = [helloworld_pb2.Coord3(
                        x=i[0], y=i[1], z=i[2]) for i in input_coords]),
                    tx=100, ty=200, tz=300
                )
            )
            end = time.time()
            print("time elapsed: %d" % (end - start))
            assert len(modified_coords.coords) == len(base_coords) * test_size_factor

            # print('modified_coords:')
            # print(modified_coords.coords);
            # print('original_coords:')
            # print(coord_list.coords);

        if False:
            x_values, y_values, z_values = numpy.transpose(
                base_coords * test_size_factor
            )
            start = time.time()
            modified_coords2 = stub.TranslateCoordList3(
                helloworld_pb2.TranslateCoordList3Request(
                    coord_list = helloworld_pb2.CoordList3(x = x_values, y = y_values, z = z_values),
                    tx=100, ty=200, tz=300
                )
            )
            end = time.time()
            print("time elapsed: %d" % (end - start))
            assert len(modified_coords2.x) == len(base_coords) * test_size_factor
            assert len(modified_coords2.y) == len(base_coords) * test_size_factor
            assert len(modified_coords2.z) == len(base_coords) * test_size_factor

            # print('modified_coords:')
            # print(modified_coords2)
            # print('original_coords:')
            # print(coord_list2)

        if True:
            input_coords = base_coords * test_size_factor
            def g(input):
                for i in input:
                    yield helloworld_pb2.TranslateCoord3Request(
                        coords=helloworld_pb2.Coord3(x=i[0], y=i[1], z=i[2]),
                        tx=100, ty=200, tz=300
                    )
            start = time.time()
            modified_coords2 = [i for i in stub.TranslateCoord3Stream(g(input_coords))]
            end = time.time()
            print("time elapsed: %d" % (end - start))
            assert len(modified_coords2) == len(base_coords) * test_size_factor

            # print('modified_coords:')
            # print(modified_coords2)
            # print('original_coords:')
            # print(coord_list2)


if __name__ == '__main__':
    logging.basicConfig()
    run()
