import numpy as np
import math


class HHNetwork:

  def __init__(self, _neuronsPerLayer, _Dmax):

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
    Izhikevich neuron update function. Update one layer for 1 millisecond
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
   


    # Update v using the Hodgen Hoxley model and Euler method
    for k in xrange(int(1/dt)):
       #save current value of layer voltage in variable v
       v = self.layer[i].v     

       #print v
       

       self.layer[i].am = (2.5 - 0.1*v)/(np.exp(2.5-0.1*v)-1.0)
       self.layer[i].an = (0.1-0.01*v)/(np.exp(1.0-0.1*v)-1.0)
       self.layer[i].ah = 0.07*np.exp(-v/20.0)
       self.layer[i].bn = 0.125*np.exp(-v/80.0)
       self.layer[i].bm = 4.0*np.exp(-v/18.0)
       self.layer[i].bh = 1.0/(np.exp(3.0-0.1*v) +1.0)
       
       #saving initial values for m,n and h
       self.layer[i].m += dt*(self.layer[i].am*(1.0-self.layer[i].m)-(self.layer[i].bm*self.layer[i].m))
       self.layer[i].n += dt*(self.layer[i].an*(1.0-self.layer[i].n)-(self.layer[i].bn*self.layer[i].n))
       self.layer[i].h += dt*(self.layer[i].ah*(1.0-self.layer[i].h)-(self.layer[i].bh*self.layer[i].h))    

       m = self.layer[i].m
       n = self.layer[i].n
       h = self.layer[i].h

       gNa = self.layer[i].gNa
       ENa = self.layer[i].ENa
       gL = self.layer[i].gL
       EL = self.layer[i].gL
       gK = self.layer[i].gK
       EK = self.layer[i].EK


      

       self.layer[i].v += dt*(-1.0*(gNa*(m**3.0)*h*(self.layer[i].v - ENa) + gK*(n**4.0)*(self.layer[i].v - EK) + gL*(self.layer[i].v - EL)) + self.layer[i].I)

       #print temp

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
  Layer of Hodgen Huxley neurons to be used inside an IzNetwork.
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

    self.m = np.zeros(n)
    self.n = np.zeros(n)
    self.h = np.zeros(n)

    self.S      = {}
    self.delay  = {}
    self.factor = {}

    self.am = np.zeros(n)
    self.an = np.zeros(n)
    self.ah = np.zeros(n)
    self.bm = np.zeros(n)
    self.bn = np.zeros(n)
    self.bh = np.zeros(n)



"""
am = {}
an = {}
ah = {}
bn = {}
bm = {}
bh = {}

am[n] = (2.5 - 0.1*v[n])/(np.exp(2.5-0.1*v[n])-1.0)
an[n] = (0.1-0.01*v[n])/(math.exp(1.0-0.1*v[n])-1.0)
ah[n] = 0.07*math.exp(-v[n]/20.0)
bn[n] = 0.125*math.exp(-v[n]/80.0)
bm[n] = 4.0*math.exp(-v[n]/18.0)
bh[n] = 1.0/(math.exp(3.0-0.1*v[n]) +1.0)

#saving initial values for m,n and h
self.layer[i].m[n] += dt*(am[n]*(1.0-self.layer[i].m[n])-(bm[n]*self.layer[i].m[n]))
self.layer[i].n[n] += dt*(an[n]*(1.0-self.layer[i].n[n])-(bn[n]*self.layer[i].n[n]))
self.layer[i].h[n] += dt*(ah[n]*(1.0-self.layer[i].h[n])-(bh[n]*self.layer[i].h[n]))      
"""
