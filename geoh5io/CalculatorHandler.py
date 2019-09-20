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

from . import interfaces


class CalculatorHandler:
    def __init__(self):
        self.log = {}

    def ping(self):
        print("ping()")

    def add(self, n1, n2):
        print("add(%d,%d)" % (n1, n2))
        return n1 + n2

    def calculate(self, logid, work):
        print("calculate(%d, %r)" % (logid, work))

        if work.op == interfaces.tutorial.Operation.ADD:
            val = work.num1 + work.num2
        elif work.op == interfaces.tutorial.Operation.SUBTRACT:
            val = work.num1 - work.num2
        elif work.op == interfaces.tutorial.Operation.MULTIPLY:
            val = work.num1 * work.num2
        elif work.op == interfaces.tutorial.Operation.DIVIDE:
            if work.num2 == 0:
                x = interfaces.tutorial.InvalidOperation()
                x.whatOp = work.op
                x.why = "Cannot divide by 0"
                raise x
            val = work.num1 / work.num2
        else:
            x = interfaces.tutorial.InvalidOperation()
            x.whatOp = work.op
            x.why = "Invalid operation"
            raise x

        log = interfaces.shared.SharedStruct()
        log.key = logid
        log.value = "%d" % (val)
        self.log[logid] = log

        return val

    def shift(self, coord_list, tx, ty, tz):
        print("shift([...],%d,%d,%d)" % (tx, ty, tz))
        return interfaces.tutorial.CoordList(
            [
                [coord[0] + tx, coord[1] + ty, coord[2] + tz]
                for coord in coord_list.coords
            ]
        )

    def shift2(self, coord_list, tx, ty, tz):
        print("shift([...],%d,%d,%d)" % (tx, ty, tz))
        coords = interfaces.tutorial.CoordList2()
        coords.x = [coord + tx for coord in coord_list.x]
        coords.y = [coord + ty for coord in coord_list.y]
        coords.z = [coord + tz for coord in coord_list.z]
        return coords

    def getStruct(self, key):
        print("getStruct(%d)" % (key))
        return self.log[key]

    def zip(self):
        print("zip()")
