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

  F    = 50.0/np.sqrt(Ns)  # Scaling factor
  D    = 4                 # Conduction delay
  Dmax = 5                 # Maximum conduction delay

  net = HHNetwork([Ns, Ns, Nm, Nm], Dmax)

  # Layer 0 (Left sensory neurons)
  r = rn.rand(Ns)
  net.layer[0].N = Ns

  # Layer 1 (Right sensory neurons)
  r = rn.rand(Ns)
  net.layer[1].N = Ns

  # Layer 2 (Left motor neurons)
  r = rn.rand(Nm)
  net.layer[2].N = Nm

  # Layer 3 (Right motor neurons)
  r = rn.rand(Nm)
  net.layer[3].N = Nm

  for i in range(0, 4):
    net.layer[i].gNa = 555.0 + 445*(r**2)
    net.layer[i].gK = 21.0 +16*(r**2)
    net.layer[i].gL = 0.075 +0.275*(r**2)
    net.layer[i].ENa = 655.5 + 544.5*(r**2)
    net.layer[i].EK = -6.0 + 8*(r**2)
    net.layer[i].EL = 92.5 +87.5*(r**2)
    net.layer[i].C = 2.5 + 2.5*(r**2)

    net.layer[i].Ik = 0.0*np.ones(Ns)
    net.layer[i].m = 0.0*np.ones(Ns)
    net.layer[i].n = 0.0*np.ones(Ns)
    net.layer[i].h = 0.0*np.ones(Ns)
    net.layer[i].alpham = 0.0*np.ones(Ns)
    net.layer[i].alphan = 0.0*np.ones(Ns)
    net.layer[i].alphah = 0.0*np.ones(Ns)
    net.layer[i].betam = 0.0*np.ones(Ns)
    net.layer[i].betan = 0.0*np.ones(Ns)
    net.layer[i].betah = 0.0*np.ones(Ns)

    """
    net.layer[i].gNa = 555.0 + 445*(r**2)
    net.layer[i].gK = 21.0 +16*(r**2)
    net.layer[i].gL = 0.075 +0.275*(r**2)
    net.layer[i].ENa = 655.5 + 544.5*(r**2)
    net.layer[i].EK = -6.0 + 8*(r**2)
    net.layer[i].EL = 92.5 +87.5*(r**2)
    net.layer[i].C = 2.5 + 2.5*(r**2)
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
