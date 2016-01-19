"""
Computational Neurodynamics
Exercise 2
(C) Murray Shanahan et al, 2015
"""

from HHNetwork import HHNetwork
import numpy as np
import numpy.random as rn


def ConnectHH2L(N0, N1):
  """
  Constructs two layers of Izhikevich neurons and connects them together.
  Layers are arrays of N neurons. Parameters for regular spiking neurons
  extracted from:
  http://www.izhikevich.org/publications/spikes.htm
  """

  F = 60/np.sqrt(N1)  # Scaling factor
  D = 5               # Conduction delay
  Dmax = 10           # Maximum conduction delay

  net = HHNetwork([N0, N1], Dmax)

  # Neuron parameters
  # Each layer comprises a heterogenous set of neurons, with a small spread
  # of parameter values, so that they exhibit some dynamical variation
  # (To get a homogenous population of canonical "regular spiking" neurons,
  # multiply r by zero.)

  # Layer 0 (Left sensory neurons)
  r = rn.rand(N0)
  net.layer[0].N = N0
  net.layer[0].gNa = 120.0 * np.ones(N0)
  net.layer[0].gK = 36.0 * np.ones(N0)
  net.layer[0].gL = 0.3 + 1*(r**2) * np.ones(N0)
  net.layer[0].ENa = 115.0 * np.ones(N0)
  net.layer[0].EK = -12.0 * np.ones(N0)
  net.layer[0].EL = 0.6  - 1*(r**2) * np.ones(N0)

  # Layer 1 (Right sensory neurons)
  r = rn.rand(N1) 
  net.layer[1].N = N1
  net.layer[1].gNa = 120.0 * np.ones(N1)
  net.layer[1].gK = 36.0 * np.ones(N1)
  net.layer[1].gL = 0.3 + 1*(r**2) * np.ones(N1)
  net.layer[1].ENa = 115.0* np.ones(N1)
  net.layer[1].EK = -12.0* np.ones(N1)
  net.layer[1].EL = 0.6 - 1*(r**2) * np.ones(N1)

  ## Connectivity matrix (synaptic weights)
  # layer[i].S[j] is the connectivity matrix from layer j to layer i
  # S(i,j) is the strength of the connection from neuron j to neuron i
  net.layer[1].S[0]      = np.ones([N1, N0])
  net.layer[1].factor[0] = F
  net.layer[1].delay[0]  = D * np.ones([N1, N0], dtype=int)

  return net
