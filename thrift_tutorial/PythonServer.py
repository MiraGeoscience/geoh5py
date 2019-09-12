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

import thriftpy2
thrift_module = thriftpy2.load("tutorial.thrift", module_name="tutorial_thrift")

from thriftpy2.rpc import make_server

import CalculatorHandler
from tutorial_thrift import Calculator, InvalidOperation, Operation


def start():
    handler = CalculatorHandler.CalculatorHandler()

    timeout_seconds = 5*60
    # TODO: get server address, port and timeout from environment variables
    server = make_server(Calculator, handler, 'localhost', 9090, client_timeout=1000*timeout_seconds)

    print('Starting the server...')
    server.serve()
    print('done.')

if __name__ == '__main__':
    start()