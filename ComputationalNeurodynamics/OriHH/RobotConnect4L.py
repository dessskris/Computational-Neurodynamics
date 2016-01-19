"""
Computational Neurodynamics
Exercise 2

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import numpy.random as rn
from HHNetwork import HHNetwork


def RobotConnect4L(Ns, Nm):
  """
  Construct four layers of Izhikevich neurons and connect them together.
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
  net.layer[0].gNa = 120.0*np.ones(Ns)
  net.layer[0].gK = 36.0*np.ones(Ns)
  net.layer[0].gL = 0.3 + 1*(r**2)
  net.layer[0].ENa = 115.0
  net.layer[0].EK = -12.0
  net.layer[0].EL = 0.6 - 1*(r**2)

  # Layer 1 (Right sensory neurons)
  r = rn.rand(Ns)
  net.layer[1].N = Ns
  net.layer[1].gNa = 120.0*np.ones(Ns)
  net.layer[1].gK = 36.0*np.ones(Ns)
  net.layer[1].gL = 0.3 + 1*(r**2)
  net.layer[1].ENa = 115.0
  net.layer[1].EK = -12.0
  net.layer[1].EL = 0.6 - 1*(r**2)

  # Layer 2 (Left motor neurons)
  r = rn.rand(Nm)
  net.layer[2].N = Nm
  net.layer[2].gNa = 120.0*np.ones(Ns)
  net.layer[2].gK = 36.0*np.ones(Ns)
  net.layer[2].gL = 0.3 + 1*(r**2)
  net.layer[2].ENa = 115.0
  net.layer[2].EK = -12.0
  net.layer[2].EL = 0.6 - 1*(r**2)

  # Layer 3 (Right motor neurons)
  r = rn.rand(Nm)
  net.layer[3].N = Nm
  net.layer[3].gNa = 120.0*np.ones(Ns)
  net.layer[3].gK = 36.0*np.ones(Ns)
  net.layer[3].gL = 0.3 + 1*(r**2)
  net.layer[3].ENa = 115.0
  net.layer[3].EK = -12.0
  net.layer[3].EL = 0.6 - 1*(r**2)


  """
  net.layer[i].gNa = 555.0 + 445*(r**2)
  net.layer[i].gK = 21.0 +16*(r**2)
  net.layer[i].gL = 0.075 +0.275*(r**2)
  net.layer[i].ENa = 655.5 + 544.5*(r**2)
  net.layer[i].EK = -6.0 + 8*(r**2)
  net.layer[i].EL = 92.5 +87.5*(r**2)
  net.layer[i].C = 2.5 + 2.5*(r**2)

  net.layer[0].gNa = 120.0*np.ones(Ns)
  net.layer[0].gK = 36.0*np.ones(Ns)
  net.layer[0].gL = 0.3 + (r**2)
  net.layer[0].ENa = 115.0*np.ones(Ns)
  net.layer[0].EK = -12.0*np.ones(Ns)
  net.layer[0].EL = 0.6 - (r**2)
  net.layer[0].C = 1*np.ones(Ns)
  """

  # Connectivity matrix (synaptic weights)
  # layer[i].S[j] is the connectivity matrix from layer j to layer i
  # s[i,j] is the streght of the connection from neuron j to neuron i

  # Connect 0 to 3 and 1 to 2 for seeking behaviour
  net.layer[3].S[0]      = np.ones([Nm, Ns])
  net.layer[3].factor[0] = F
  net.layer[3].delay[0]  = D * np.ones([Nm, Ns], dtype=int)

  net.layer[2].S[1]      = np.ones([Nm, Ns])
  net.layer[2].factor[1] = F
  net.layer[2].delay[1]  = D * np.ones([Nm, Ns], dtype=int)

  return net
