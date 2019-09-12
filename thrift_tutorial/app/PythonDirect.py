#!/usr/bin/env python3

import sys
sys.path.append('gen-py')

import time

from app import interfaces

from app import CalculatorHandler

def main():

    # Create a client to use the protocol encoder
    client = CalculatorHandler.CalculatorHandler()

    client.ping()
    print('ping()')

    sum_ = client.add(1, 1)
    print('1+1=%d' % sum_)

    work = interfaces.tutorial.Work()

    work.op = interfaces.tutorial.Operation.DIVIDE
    work.num1 = 1
    work.num2 = 0

    try:
        quotient = client.calculate(1, work)
        print('Whoa? You know how to divide by zero?')
        print('FYI the answer is %d' % quotient)
    except interfaces.tutorial.InvalidOperation as e:
        print('InvalidOperation: %r' % e)

    test_size_factor = 3 * 10000
    base_coords = [[0,1,2],[10,11,12],[20,21,22],[30,31,32]]
    coord_list = interfaces.tutorial.CoordList(base_coords * test_size_factor)
    start = time.time()
    modified_coords = client.shift(coord_list, 100, 200, 300)
    end = time.time()
    assert( len(modified_coords.coords) == len(base_coords) * test_size_factor)
    print('time elapsed: %d' % (end - start))

    #print('modified_coords:')
    #print(modified_coords.coords);    
    #print('original_coords:')
    #print(coord_list.coords);
    
    work.op = interfaces.tutorial.Operation.SUBTRACT
    work.num1 = 15
    work.num2 = 10

    diff = client.calculate(1, work)
    print('15-10=%d' % diff)

    log = client.getStruct(1)
    print('Check log: %s' % log.value)


if __name__ == '__main__':
    main()
