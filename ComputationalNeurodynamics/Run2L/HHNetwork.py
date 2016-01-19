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
    dt = 0.2

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
      self.layer[i].alphan = (0.1 - 0.01*self.layer[i].v)/(np.exp(1.0 - 0.1*self.layer[i].v)-1)
      self.layer[i].alpham = (2.5 - 0.1*self.layer[i].v)/(np.exp(2.5 - 0.1*self.layer[i].v)-1)
      self.layer[i].alphah = 0.07*np.exp(-self.layer[i].v/20.0)

      self.layer[i].betan = 0.125*np.exp(-self.layer[i].v/80.0)
      self.layer[i].betam = 4.0*np.exp(-self.layer[i].v/18.0)
      self.layer[i].betah = 1.0/(np.exp(3.0-0.1*self.layer[i].v)+1.0)

      self.layer[i].m += dt*(self.layer[i].alpham*(1-self.layer[i].m) - self.layer[i].betam*self.layer[i].m)
      self.layer[i].n += dt*(self.layer[i].alphan*(1-self.layer[i].n) - self.layer[i].betan*self.layer[i].n)
      self.layer[i].h += dt*(self.layer[i].alphah*(1-self.layer[i].h) - self.layer[i].betah*self.layer[i].h)

      self.layer[i].Ik = self.layer[i].gNa*(self.layer[i].m**3)*self.layer[i].h*(self.layer[i].v-self.layer[i].ENa) + self.layer[i].gK*(self.layer[i].n**4)*(self.layer[i].v-self.layer[i].EK) + self.layer[i].gL*(self.layer[i].v-self.layer[i].EL)

      self.layer[i].v += dt*(-self.layer[i].Ik + self.layer[i].I)/self.layer[i].C

      # Find index of neurons that have fired this millisecond
      fired = np.where(self.layer[i].v >= 30)[0]

      if len(fired) > 0:
        for f in fired:
          # Add spikes into spike train
          if len(self.layer[i].firings) != 0:
            self.layer[i].firings = np.vstack([self.layer[i].firings, [t, f]])
          else:
            self.layer[i].firings = np.array([[t, f]])

          # Reset the membrane potential after spikes
          self.layer[i].v[f]  = 0

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
    self.Ik = np.zeros(n)
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
