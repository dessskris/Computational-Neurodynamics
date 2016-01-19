"""
Computational Neurodynamics
Exercise 2
(C) Murray Shanahan et al, 2015
"""

import numpy as np
import numpy.random as rn
from HHNetwork import HHNetwork


def RobotHHConnect4L(Ns, Nm):
  """
  Construct four layers of Hodgen Huxley neurons and connect them together.
  Layers 0 and 1 comprise sensory neurons, while layers 2 and 3 comprise
  motor neurons. Sensory neurons excite contralateral motor neurons causing
  seeking behaviour. Layers are heterogenous populations of Izhikevich
  neurons with slightly different parameter values.
  Inputs:
  Ns -- Number of neurons in sensory layers
  Nm -- Number of neurons in motor layers
  """

  F    = 20.0/np.sqrt(Ns)  # Scaling factor
  D    = 4                 # Conduction delay
  Dmax = 5                 # Maximum conduction delay

  net = HHNetwork([Ns, Ns, Nm, Nm], Dmax)

  # Layer 0 (Left sensory neurons)
  r = rn.rand(Ns)
  net.layer[0].N = Ns
  net.layer[0].gNa = 120.0 * np.ones(Ns)
  net.layer[0].gK = 36.0 * np.ones(Ns)
  net.layer[0].gL = 0.3 + 1*(r**2)
  net.layer[0].ENa = 115.0
  net.layer[0].EK = -12.0
  net.layer[0].EL = 0.6 - 1*(r**2)

  # Layer 1 (Right sensory neurons)
  r = rn.rand(Ns) 
  net.layer[1].N = Ns
  net.layer[1].gNa = 120.0 * np.ones(Ns)
  net.layer[1].gK = 36.0 * np.ones(Ns)
  net.layer[1].gL = 0.3 + 1*(r**2)
  net.layer[1].ENa = 115.0
  net.layer[1].EK = -12.0
  net.layer[1].EL = 0.6 - 1*(r**2)

 # Layer 2 (Left motor neurons)
  r = rn.rand(Ns)
  net.layer[2].N = Nm
  net.layer[2].gNa = 120.0 * np.ones(Ns)
  net.layer[2].gK = 36.0 * np.ones(Ns)
  net.layer[2].gL = 0.3 + 1*(r**2)
  net.layer[2].ENa = 115.0
  net.layer[2].EK = -12.0
  net.layer[2].EL = 0.6 - 1*(r**2)


  # Layer 3 (Right motor neurons)
  r = rn.rand(Ns)
  net.layer[3].N = Nm
  net.layer[3].gNa = 120.0 * np.ones(Ns)
  net.layer[3].gK = 36.0 * np.ones(Ns)
  net.layer[3].gL = 0.3 + 1*(r**2)
  net.layer[3].ENa = 115.0
  net.layer[3].EK = -12.0
  net.layer[3].EL = 0.6 - 1*(r**2)


  """
  # Layer 1 (Left neurons inhibitory)
  r = rn.rand(Ns)
  net.layer[4].N = Ns
  net.layer[4].a = 0.002 * np.ones(Ns)
  net.layer[4].b = 0.25 * np.ones(Ns)
  net.layer[4].c = -65 + 15*(r**2)
  net.layer[4].d = 2 - 6*(r**2)

  # Layer 3 (Right neurons inhibitory)
  r = rn.rand(Ns)
  net.layer[5].N = Ns
  net.layer[5].a = 0.002 * np.ones(Ns)
  net.layer[5].b = 0.25 * np.ones(Ns)
  net.layer[5].c = -65 + 15*(r**2)
  net.layer[5].d = 2 - 6*(r**2)
  """
 

  # Connectivity matrix (synaptic weights)
  # layer[i].S[j] is the connectivity matrix from layer j to layer i
  # s[i,j] is the strenght of the connection from neuron j to neuron i

  # Connect 0 to 2 and 1 to 3 for seeking behaviour
  net.layer[2].S[1]      = np.ones([Nm, Ns])
  net.layer[2].factor[1] = F
  net.layer[2].delay[1]  = D * np.ones([Nm, Ns], dtype=int)

  net.layer[3].S[0]      = np.ones([Nm, Ns])
  net.layer[3].factor[0] = F
  net.layer[3].delay[0]  = D * np.ones([Nm, Ns], dtype=int)

  """
  # Connect 2 to 4, and 3 to 5, to excite matching inhibitory population
  
  net.layer[4].S[2]      = np.ones([Nm, Ns])
  net.layer[4].factor[2] = F
  net.layer[4].delay[2]  = D * np.ones([Nm, Ns], dtype=int)  # no delay for inhibition?

  net.layer[5].S[3]      = np.ones([Nm, Ns])
  net.layer[5].factor[3] = F
  net.layer[5].delay[3]  = D * np.ones([Nm, Ns], dtype=int)  # no delay for inhibition?


 # Connect 2 to 5, and 3 to 4, to inhibit competitor
  
  net.layer[3].S[4]      = -1 * np.ones([Nm, Ns])
  net.layer[3].factor[4] = F
  net.layer[3].delay[4]  = D * np.ones([Nm, Ns], dtype=int)  # no delay for inhibition?

  net.layer[2].S[5]      = -1 * np.ones([Nm, Ns])
  net.layer[2].factor[5] = F
  net.layer[2].delay[5]  = D * np.ones([Nm, Ns], dtype=int)  # no delay for inhibition?


  # Connect 3 to 4, and 4 to 3, to inhibit competitor
  
  net.layer[4].S[5]      = -1 * np.ones([Nm, Ns])
  net.layer[4].factor[5] = F
  net.layer[4].delay[5]  = D * np.ones([Nm, Ns], dtype=int)  # no delay for inhibition?

  net.layer[5].S[4]      = -1 * np.ones([Nm, Ns])
  net.layer[5].factor[4] = F
  net.layer[5].delay[4]  = D * np.ones([Nm, Ns], dtype=int)  # no delay for inhibition?
"""

  return net
