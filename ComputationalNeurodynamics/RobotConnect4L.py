"""
Computational Neurodynamics
Exercise 2

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import numpy.random as rn
from IzNetwork import IzNetwork


def RobotConnect4L(Ns, Nm, Ni):
  """
  Construct six layers of Izhikevich neurons and connect them together.
  Layers 0 and 1 comprise sensory neurons, while layers 2 and 3 comprise
  motor neurons, and layers 4 and 5 comprise inhibitory neurons. Sensory
  neurons excite contralateral motor neurons causing
  seeking behaviour. Layers are heterogenous populations of Izhikevich
  neurons with slightly different parameter values.

  Inputs:
  Ns -- Number of neurons in sensory layers
  Nm -- Number of neurons in motor layers
  Ni -- Number of neurons in inhibitory layers
  """

  F    = 50.0/np.sqrt(Ns)  # Scaling factor
  D    = 4                 # Conduction delay
  Dmax = 5                 # Maximum conduction delay

  net = IzNetwork([Ns, Ns, Nm, Nm, Ni, Ni], Dmax)

  # Layer 0 (Left sensory neurons)
  r = rn.rand(Ns)
  net.layer[0].N = Ns
  net.layer[0].a = 0.02 * np.ones(Ns)
  net.layer[0].b = 0.20 * np.ones(Ns)
  net.layer[0].c = -65 + 15*(r**2)
  net.layer[0].d = 8 - 6*(r**2)

  # Layer 1 (Right sensory neurons)
  r = rn.rand(Ns)
  net.layer[1].N = Ns
  net.layer[1].a = 0.02 * np.ones(Ns)
  net.layer[1].b = 0.20 * np.ones(Ns)
  net.layer[1].c = -65 + 15*(r**2)
  net.layer[1].d = 8 - 6*(r**2)

  # Layer 2 (Left motor neurons)
  r = rn.rand(Nm)
  net.layer[2].N = Nm
  net.layer[2].a = 0.02 * np.ones(Nm)
  net.layer[2].b = 0.20 * np.ones(Nm)
  net.layer[2].c = -65 + 15*(r**2)
  net.layer[2].d = 8 - 6*(r**2)

  # Layer 3 (Right motor neurons)
  r = rn.rand(Nm)
  net.layer[3].N = Nm
  net.layer[3].a = 0.02 * np.ones(Nm)
  net.layer[3].b = 0.20 * np.ones(Nm)
  net.layer[3].c = -65 + 15*(r**2)
  net.layer[3].d = 8 - 6*(r**2)

  # Layer 4 (Left inhibitory neurons)
  r = rn.rand(Ni)
  net.layer[4].N = Ni
  net.layer[4].a = 0.02 * np.ones(Nm)
  net.layer[4].b = 0.20 * np.ones(Nm)
  net.layer[4].c = -65 + 15*(r**2)
  net.layer[4].d = 8 - 6*(r**2)

  # Layer 5 (Right inhibitory neurons)
  r = rn.rand(Ni)
  net.layer[5].N = Ni
  net.layer[5].a = 0.02 * np.ones(Nm)
  net.layer[5].b = 0.20 * np.ones(Nm)
  net.layer[5].c = -65 + 15*(r**2)
  net.layer[5].d = 8 - 6*(r**2)

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

  # Connect 2 to 4 and 3 to 5
  net.layer[4].S[2]      = np.ones([Ni, Nm])
  net.layer[4].factor[2] = F
  net.layer[4].delay[2]  = D * np.ones([Ni, Nm], dtype=int)

  net.layer[5].S[3]      = np.ones([Ni, Nm])
  net.layer[5].factor[3] = F
  net.layer[5].delay[3]  = D * np.ones([Ni, Nm], dtype=int)

  # Connect 2 to 5 and 3 to 4
  net.layer[3].S[4]      = (-1)*np.ones([Ni, Nm])
  net.layer[3].factor[4] = F
  net.layer[3].delay[4]  = D * np.ones([Ni, Nm], dtype=int)

  net.layer[2].S[5]      = (-1)*np.ones([Ni, Nm])
  net.layer[2].factor[5] = F
  net.layer[2].delay[5]  = D * np.ones([Ni, Nm], dtype=int)

  # Connect 4 and 5 (both ways)
  net.layer[4].S[5]      = (-1)*np.ones([Ni, Ni])
  net.layer[4].factor[5] = F
  net.layer[4].delay[5]  = D * np.ones([Ni, Ni], dtype=int)

  net.layer[5].S[4]      = (-1)*np.ones([Ni, Ni])
  net.layer[5].factor[4] = F
  net.layer[5].delay[4]  = D * np.ones([Ni, Ni], dtype=int)


  return net
