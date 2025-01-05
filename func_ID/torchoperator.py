
import numpy as np
from _torchoperator import _TorchOperator
from _torchoperator_list import _TorchOperator_list

class TorchOperator():
    """Wrap a function into a Torch function.
    """
    def __init__(
        self,
        Op,
        Op_H,
        device,
        devicetorch,
        ):
            
            self.matvec  = Op
            self.rmatvec = Op_H
            self.device  = device
            self.devicetorch = devicetorch
            self.Top    = _TorchOperator.apply

    def apply(self, x):
        """Apply forward pass to input vector
        """
        return self.Top(x, self.matvec, self.rmatvec, self.device, self.devicetorch)


class TorchOperator_list():
    """Wrap a function into a Torch function.
    """
    def __init__(
        self,
        Op,
        Op_H,
        device,
        devicetorch,
        ):
            
            self.matvec  = Op
            self.rmatvec = Op_H
            self.device  = device
            self.devicetorch = devicetorch
            self.Top    = _TorchOperator_list.apply

    def apply(self, x):
        """Apply forward pass to input vector
        """
        return self.Top(x, self.matvec, self.rmatvec, self.device, self.devicetorch)