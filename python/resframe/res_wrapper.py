"""
This module is responsible for the initialization of the reservoire, and a useful abstraction to 
the learning module. The api that is going to be exposed is somewhat volatile for now.
"""

from typing import List, Tuple, Optional, Union
from numpy import random
import resframe


class ResFrameNewModel:
    # wrapper for the model setup
    def __init__(
        self,
        a: List[float],
        b: List[float],
        c: List[float],
        d: List[float],
        v: List[float],
        u: List[float],
        connections: List[List[float]]
    ):
        self.reservoire = resframe.NewModel(a, b, c, d, v, u, connections)

    # ensures that the reservoire is constructible in case some parameters are not set
    def from_n(
        self, 
        n: int,
        a: Union[List[float], None] = None,
        b: Union[List[float], None] = None,
        c: Union[List[float], None] = None,
        d: Union[List[float], None] = None,
        v: Union[List[float], None] = None,
        u: Union[List[float], None] = None,
        connections: Union[List[List[float]], None] = None,
    ):
        a = a or [0.02] * n
        b = b or [0.2] * n
        c = c or [-65.] * n
        d = d or [8.] * n
        v = v or [-65.] * n
        u = u or [0.2 * -64.] * n
        connections = connections or [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
        self.__init__(a, b, c, d, v, u, connections)

    def step(self, input: List[float], dt: float) -> List[float]:
        input = self.reservoire.diffuse(input)
        return self.reservoire.excite(input, dt)

