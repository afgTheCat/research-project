import enum
from typing import List, Tuple, Optional


def reservoire_test() -> Tuple[List[float], List[List[float]]]: ...
def test_frame() -> None: ...

class NetworkInitPrimitive(enum.Enum):
    NoRandomWeight = '1',
    NormalRandomWeight = '2'

class ConnectivityPrimitive(enum.Enum):
    ErdosUniform = '1',
    ErdosNormal = '2',



class InputSteps():
    def __init__(self, input_vals: List[Tuple[float, List[float]]]) -> None: ...

class VariantChooser():
    def __init__(self,
                 network_init_primitive = NetworkInitPrimitive.NoRandomWeight,
                 connectivity_primitive = ConnectivityPrimitive.ErdosUniform,
                 network_membrane_potential = -65.0,
                 network_recovery_variable = -14.0,
                 network_membrane_potential_dev = 0.0,
                 network_recovery_variable_dev = 0.0,
                 erdos_connectivity = 1.0,
                 erdos_uniform_lower = 0.0,
                 erdos_uniform_upper = 1.0,
                 erdos_normal_mean = 0.0,
                 erdos_normal_dev = 1.0
    ) -> None: ...

class Reservoire():
    def __init__(self,
                 a: Optional[float] = 0.02,
                 b: Optional[float] = 0.2,
                 c: Optional[float] = -65.0,
                 d: Optional[float] = 8.0,
                 dt: Optional[float] = 0.05,
                 spike_val: Optional[float] = 35.0,
                 number_of_neurons: Optional[int] = 10,
                 variant_chooser: Optional[VariantChooser] = VariantChooser()
    ) -> None: ...
    def get_states(self, input: InputSteps) -> Optional[Tuple[List[float], List[List[float]]]]: ...

class RCModel():
    def __init__(self) -> None: ...
    def test_reservoire(self) -> None: ...
