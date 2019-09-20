#!/usr/bin/env python3

#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
#

import time

import numpy
from geoh5io import interfaces
from thriftpy2.rpc import make_client


def main():

    timeout_seconds = 5 * 60
    # TODO: get server address, port and timeout from environment variables
    client = make_client(
        interfaces.tutorial.Calculator,
        "localhost",
        9090,
        timeout=1000 * timeout_seconds,
    )

    client.ping()
    print("ping()")

    sum_ = client.add(1, 1)
    print("1+1=%d" % sum_)

    work = interfaces.tutorial.Work()

    work.op = interfaces.tutorial.Operation.DIVIDE
    work.num1 = 1
    work.num2 = 0

    try:
        quotient = client.calculate(1, work)
        print("Whoa? You know how to divide by zero?")
        print("FYI the answer is %d" % quotient)
    except interfaces.tutorial.InvalidOperation as e:
        print("InvalidOperation: %r" % e)

    test_size_factor = 3 * int(1e4)
    # CoordList v/s CoordList2 v/s no client
    # x 1e4 takes 1  / 0 / 0 second
    # x 1e5 takes 17 / 8 / 0 seconds
    # x 1e6 takes ~150 / 83 / 2 seconds

    test_list2 = True
    base_coords = [[0, 1, 2], [10, 11, 12], [20, 21, 22], [30, 31, 32]]
    if not test_list2:
        coord_list = interfaces.tutorial.CoordList(base_coords * test_size_factor)
        start = time.time()
        modified_coords = client.shift(coord_list, 100, 200, 300)
        end = time.time()
        assert len(modified_coords.coords) == len(base_coords) * test_size_factor
        print("time elapsed: %d" % (end - start))

        # print('modified_coords:')
        # print(modified_coords.coords);
        # print('original_coords:')
        # print(coord_list.coords);

    else:
        coord_list2 = interfaces.tutorial.CoordList2()
        coord_list2.x, coord_list2.y, coord_list2.z = numpy.transpose(
            base_coords * test_size_factor
        )
        start = time.time()
        modified_coords2 = client.shift2(coord_list2, 100, 200, 300)
        end = time.time()
        assert len(modified_coords2.x) == len(base_coords) * test_size_factor
        assert len(modified_coords2.y) == len(base_coords) * test_size_factor
        assert len(modified_coords2.z) == len(base_coords) * test_size_factor
        print("time elapsed: %d" % (end - start))

        # print('modified_coords:')
        # print(modified_coords2)
        # print('original_coords:')
        # print(coord_list2)

    work.op = interfaces.tutorial.Operation.SUBTRACT
    work.num1 = 15
    work.num2 = 10

    diff = client.calculate(1, work)
    print("15-10=%d" % diff)

    log = client.getStruct(1)
    print("Check log: %s" % log.value)

    client.close()


if __name__ == "__main__":
    main()
