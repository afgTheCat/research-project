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

class InputStepsHomogenous():
    def __init__(self, input_vals: List[Tuple[float, List[float]]]) -> None: ...
    def vals(self) -> List[List[float]]: ...

class InputStepsHeterogenous():
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

class HomogenousReservoire():
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
    def get_states(self, input: InputStepsHomogenous) -> Tuple[List[float], List[List[float]]]: ...

class HeterogenousReservoire():
    def __init__(self, number_of_neurons: int, dt: float) -> None: ...
    def get_states(self, input: InputStepsHomogenous) -> Tuple[List[float], List[List[float]]]: ...


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
    def create_reservoire_input(self, X) -> List[InputStepsHomogenous]: ...
    def reservoire_states(self, input: List[InputStepsHomogenous]) -> Tuple[float, np.ndarray]: ...
    def reservoire_states_with_times(self, input: List[InputStepsHomogenous]) -> np.ndarray: ...
    def train(self, input: List[InputStepsHomogenous], labels) -> None: ...
    def test(self, input: List[InputStepsHomogenous], labels) -> Tuple[float, float]: ...

class RCModelHeterogenous():
    def __init__(self,
        number_of_neurons,
        dt,
        representation="last",
        readout="lin",
        w_ridge=5,
        w_ridge_embedding=10.0,
        input_delay=0.0,
        input_bias=0,
        input_scale=10,
        n_drop=0,
        ) -> None: ...
    def test_reservoire(self) -> None: ...
    def create_reservoire_input(self, X) -> List[InputStepsHomogenous]: ...
    def reservoire_states(self, input: List[InputStepsHomogenous]) -> Tuple[float, np.ndarray]: ...
    def reservoire_states_with_times(self, input: List[InputStepsHomogenous]) -> np.ndarray: ...
    def train(self, input: List[InputStepsHomogenous], labels) -> None: ...
    def test(self, input: List[InputStepsHomogenous], labels) -> Tuple[float, float]: ...
    


class NewModel():
    def __init__(
        self,
        a: List[float],
        b: List[float],
        c: List[float],
        d: List[float],
        v: List[float],
        u: List[float],
        connections: List[List[float]]
    ) -> None: ...
    def diffuse(self, input: List[float]) -> List[float]: ...
    def excite(self, input: List[float], dt: float) -> List[float]: ...

