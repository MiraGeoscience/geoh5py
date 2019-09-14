from dataclasses import dataclass
from enum import IntEnum
from typing import *












@dataclass
class SharedStruct:
    key: Optional[int] = None
    value: Optional[str] = None




class SharedService:
    def getStruct(
        self,
        key: int,
    ) -> SharedStruct:
        ...
