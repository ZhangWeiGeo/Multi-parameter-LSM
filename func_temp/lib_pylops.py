from pylops import LinearOperator
from pylops.utils                      import dottest
from pylops.utils.wavelets             import *
from pylops.utils.seismicevents        import *
from pylops.utils.tapers               import *
from pylops.basicoperators             import *
from pylops.signalprocessing           import *
from pylops.waveeqprocessing.kirchhoff import Kirchhoff
from pylops.avo.poststack              import PoststackLinearModelling, PoststackInversion

from pylops.optimization.leastsquares  import *
from pylops.optimization.sparsity      import *

from pyproximal.proximal import *
from pyproximal.optimization.primal import *
from pyproximal.optimization.primaldual import *
from pylops import TorchOperator

#########pylops
# d1_0_op             = FirstDerivative( dims=list(ini_inv[0].shape), axis=0, dtype=np.float32)
# d1_1_op             = FirstDerivative( dims=list(ini_inv[0].shape), axis=1, dtype=np.float32)

# d1_0_op_torch       = TorchOperator(d1_0_op,   device=device)
# d1_0_op_torch_H     = TorchOperator(d1_0_op.H, device=device)

# d1_1_op_torch       = TorchOperator(d1_1_op,   device=device)
# d1_1_op_torch_H     = TorchOperator(d1_1_op.H, device=device)