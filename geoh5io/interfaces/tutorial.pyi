from dataclasses import dataclass
from enum import IntEnum
from typing import *


from . import shared




class InvalidOperation(Exception):
    whatOp: Optional[int] = None
    why: Optional[str] = None




class Operation(IntEnum):
    ADD = 1
    SUBTRACT = 2
    MULTIPLY = 3
    DIVIDE = 4




@dataclass
class Work:
    num1: Optional[int] = 0
    num2: Optional[int] = None
    op: Optional[int] = None
    comment: Optional[str] = None

@dataclass
class CoordList:
    coords: Optional[List[List[float]]] = None




class Calculator:
    def ping(
        self,
    ) -> None:
        ...
    def add(
        self,
        num1: int,
        num2: int,
    ) -> int:
        ...
    def calculate(
        self,
        logid: int,
        w: Work,
    ) -> int:
        ...
    def shift(
        self,
        coord_list: CoordList,
        tx: float,
        ty: float,
        tz: float,
    ) -> CoordList:
        ...
    def zip(
        self,
    ) -> None:
        ...
    def getStruct(
        self,
        key: int,
    ) -> shared.SharedStruct:
        ...
