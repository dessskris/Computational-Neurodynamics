import numpy as np


class HHNetwork:
  """
  Network of Hodgkin Huxley neurons.
  """

  def __init__(self, _neuronsPerLayer, _Dmax):
    """
    Initialise network with given number of neurons

    Inputs:
    _neuronsPerLayer -- List with the number of neurons in each layer. A list
                        [N1, N2, ... Nk] will return a network with k layers
                        with the corresponding number of neurons in each.

    _Dmax            -- Maximum delay in all the synapses in the network. Any
                        longer delay will result in failing to deliver spikes.
    """

    self.Dmax = _Dmax
    self.Nlayers = len(_neuronsPerLayer)

    self.layer = {}

    for i, n in enumerate(_neuronsPerLayer):
      self.layer[i] = HHLayer(n)

  def Update(self, t):
    """
    Run simulation of the whole network for 1 millisecond and update the
    network's internal variables.

    Inputs:
    t -- Current timestep. Necessary to sort out the synaptic delays.
    """
    for lr in xrange(self.Nlayers):
      self.NeuronUpdate(lr, t)

  def NeuronUpdate(self, i, t):
    """
    Hodgkin Huxley neuron update function. Update one layer for 1 millisecond
    using the Euler method.

    Inputs:
    i -- Number of layer to update
    t -- Current timestep. Necessary to sort out the synaptic delays.
    """

    # Euler method step size in ms
    dt = 0.01

    # Calculate current from incoming spikes
    for j in xrange(self.Nlayers):

      # If layer[i].S[j] exists then layer[i].factor[j] and
      # layer[i].delay[j] have to exist
      if j in self.layer[i].S:
        S = self.layer[i].S[j]  # target neuron->rows, source neuron->columns

        # Firings contains time and neuron idx of each spike.
        # [t, index of the neuron in the layer j]
        firings = self.layer[j].firings

        # Find incoming spikes taking delays into account
        delay = self.layer[i].delay[j]
        F = self.layer[i].factor[j]

        # Sum current from incoming spikes
        k = len(firings)
        while k > 0 and (firings[k-1, 0] > (t - self.Dmax)):
          idx = delay[:, firings[k-1, 1]] == (t-firings[k-1, 0])
          self.layer[i].I[idx] += F * S[idx, firings[k-1, 1]]
          k = k-1

    # Update v using the HH model and Euler method
    for k in xrange(int(1/dt)):
      #print self.layer[i].v
      v = self.layer[i].v

      self.layer[i].alphan = (0.1 - 0.01*v)/(np.exp(1.0 - 0.1*v)-1.0)
      self.layer[i].alpham = (2.5 - 0.1*v)/(np.exp(2.5 - 0.1*v)-1.0)
      self.layer[i].alphah = 0.07*np.exp(-v/20.0)

      self.layer[i].betan = 0.125*np.exp(-v/80.0)
      self.layer[i].betam = 4.0*np.exp(-v/18.0)
      self.layer[i].betah = 1.0/(np.exp(3.0-0.1*v)+1.0)

      an = self.layer[i].alphan
      am = self.layer[i].alpham
      ah = self.layer[i].alphah
      bn = self.layer[i].betan
      bm = self.layer[i].betam
      bh = self.layer[i].betah

      self.layer[i].m += dt*(am*(1.0-self.layer[i].m) - (bm*self.layer[i].m))
      self.layer[i].n += dt*(an*(1.0-self.layer[i].n) - (bn*self.layer[i].n))
      self.layer[i].h += dt*(ah*(1.0-self.layer[i].h) - (bh*self.layer[i].h))

      m = self.layer[i].m
      n = self.layer[i].n
      h = self.layer[i].h

      gNa = self.layer[i].gNa
      ENa = self.layer[i].ENa
      gK = self.layer[i].gK
      EK = self.layer[i].EK
      gL = self.layer[i].gL
      EL = self.layer[i].EL


      self.layer[i].v += dt*((-1.0)*(gNa*(m**3.0)*h*(v-ENa) + gK*(n**4)*(v-EK) + gL*(v-EL)) + self.layer[i].I)

      # Find index of neurons that have fired this millisecond
      fired = np.where(self.layer[i].v >= 30)[0]

      if len(fired) > 0:
        for f in fired:
          # Add spikes into spike train
          if len(self.layer[i].firings) != 0:
            self.layer[i].firings = np.vstack([self.layer[i].firings, [t, f]])
          else:
            self.layer[i].firings = np.array([[t, f]])


    return


class HHLayer:
  """
  Layer of Hodgkin Huxley neurons to be used inside a HHNetwork.
  """

  def __init__(self, n):
    """
    Initialise layer with empty vectors.

    Inputs:
    n -- Number of neurons in the layer
    """

    self.N = n
    self.gNa = np.zeros(n)
    self.gK = np.zeros(n)
    self.gL = np.zeros(n)
    self.ENa = np.zeros(n)
    self.EK = np.zeros(n)
    self.EL = np.zeros(n)
    self.C = np.zeros(n)
    self.m = np.zeros(n)
    self.n = np.zeros(n)
    self.h = np.zeros(n)
    self.alpham = np.zeros(n)
    self.alphan = np.zeros(n)
    self.alphah = np.zeros(n)
    self.betam = np.zeros(n)
    self.betan = np.zeros(n)
    self.betah = np.zeros(n)

    self.S      = {}
    self.delay  = {}
    self.factor = {}
