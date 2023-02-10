import enum
from typing import List, Tuple, Optional
import numpy as np

def sensitivity_analysis(reservoire_params, table_name) -> None: ...
def create_connection(): ...

class InputPrimitive(enum.Enum):
    AllConnected = '1',
    PercentageConnected = '2'

class NetworkInitPrimitive(enum.Enum):
    NoRandomWeight = '1',
    NormalRandomWeight = '2'

class ConnectivityPrimitive(enum.Enum):
    ErdosUniform = '1',
    ErdosNormal = '2',
    ErdosSpectral = '3'

class ThalmicPrimitive(enum.Enum):
    Const = '1',
    Normal = '2'

class InputSteps():
    def __init__(self, input_vals: List[Tuple[float, List[float]]]) -> None: ...
    def vals(self) -> List[List[float]]: ...

class VariantChooser():
    def __init__(self,
                 network_init_primitive = NetworkInitPrimitive.NoRandomWeight,
                 connectivity_primitive = ConnectivityPrimitive.ErdosUniform,
                 input_primitive = InputPrimitive.AllConnected,
                 thalmic_primitive = ThalmicPrimitive.Const,
                 network_membrane_potential = -65.0,
                 network_recovery_variable = -14.0,
                 network_membrane_potential_dev = 0.0,
                 network_recovery_variable_dev = 0.0,
                 erdos_connectivity = 1.0,
                 erdos_uniform_lower = 0.0,
                 erdos_uniform_upper = 1.0,
                 erdos_normal_mean = 0.0,
                 erdos_normal_dev = 1.0,
                 erdos_spectral_radius = 0.59,
                 input_connectivity_p = 0.5,
                 thalmic_mean = 0.0,
                 thalmic_dev = 0.0
    ) -> None: ...

# class Reservoire():
#     def __init__(self,
#                  a: Optional[float] = 0.02,
#                  b: Optional[float] = 0.2,
#                  c: Optional[float] = -65.0,
#                  d: Optional[float] = 8.0,
#                  dt: Optional[float] = 0.05,
#                  spike_val: Optional[float] = 35.0,
#                  number_of_neurons: Optional[int] = 10,
#                  variant_chooser: Optional[VariantChooser] = VariantChooser()
#     ) -> None: ...
#     def get_states(self, input: InputSteps) -> Tuple[List[float], List[List[float]]]: ...

class RCModelHomogenous():
    def __init__(self,
                 readout = "lin",
                 w_ridge=5,
                 dt=0.05,
                 representation="last",
                 a=0.02,
                 b=0.2,
                 c=-65.0,
                 d=8.0,
                 number_of_neurons=20,
                 network_init_primitive = NetworkInitPrimitive.NoRandomWeight,
                 network_membrane_potential = -65.0,
                 network_recovery_variable = -14.0,
                 network_membrane_potential_dev = 0.0,
                 network_recovery_variable_dev = 0.0,
                 connectivity_primitive = ConnectivityPrimitive.ErdosUniform,
                 erdos_connectivity = 1.0,
                 erdos_uniform_lower = 0.0,
                 erdos_uniform_upper = 1.0,
                 erdos_normal_mean = 0.0,
                 erdos_normal_dev = 1.0,
                 erdos_spectral_radius = 0.59,
                 input_primitive=InputPrimitive.AllConnected,
                 input_connectivity_p=0.5,
                 input_bias=0,
                 input_scale=10,
                 thalmic_primitive = ThalmicPrimitive.Const,
                 thalmic_mean = 0.0,
                 thalmic_dev = 0.0
    ) -> None: ...
    def test_reservoire(self) -> None: ...
    def create_reservoire_input(self, X) -> List[InputSteps]: ...
    def reservoire_states(self, input: List[InputSteps]) -> Tuple[float, np.ndarray]: ...
    def reservoire_states_with_times(self, input: List[InputSteps]) -> np.ndarray: ...
    def train(self, input: List[InputSteps], labels) -> None: ...
    def test(self, input: List[InputSteps], labels) -> Tuple[float, float]: ...

