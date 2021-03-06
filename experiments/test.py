"""A simple test experiment definition
   
   Just a 2D normal model

   Current list of required objects:

     name
     general_model
     make_mu_model
     null_parameters
     get_seeds
     get_seeds_null
     null_options
     general_options
     observed_data
     DOF
"""

import JMCtools.distributions as jtd
import JMCtools.models as jtm
import numpy as np
import scipy.stats as sps
from functools import partial

class Experiment:
    """Just a dummy box to carry information about our experiments"""
    def __init__(self,name):
        self.name = name

name = "test"
e = Experiment(name)

# "Background" predictions
b = [5,6]

# s_MLE is not required, just useful for testing
s_MLE = [1.5,1.5]

# Create parameter mappings
# Proxy for 'signal strength' parameter added
def pars1(mu,mu1):
    return {"loc": b[0] + mu*mu1, "scale":1}

def pars2(mu,mu2):
    return {"loc": b[1] + mu*mu2, "scale":1}

# Create the joint PDF object
general_model = jtm.ParameterModel([jtd.TransDist(sps.norm,pars1),
                                    jtd.TransDist(sps.norm,pars2)])
# Create the "observed" data
# Need extra axes for matching shape of many simulated datasets
observed_data = np.array([6.5,7.5])[np.newaxis,np.newaxis,:]

# Define the null hypothesis
null_parameters = {'mu':0, 'mu1':0, 'mu2':0}

# Define functions to get good starting guesses for fitting simulated data
def get_seeds(samples):
    X1 = samples[...,0]
    X2 = samples[...,1]
    return {'mu1':X1 - b[0], 'mu2':X2 - b[1]}

# Same for fitting null hypothesis. But for this model there are no nuisance
# parameters so there is nothing to fit
def get_seeds_null(samples):
    return {}

# Set options for fit
# For null hypothesis there is nothing to fit! Let's see if we can force Minuit to
# just compute the pdf for one point for us.
null_options = {**null_parameters, 'fix_mu': True, 'fix_mu1': True, 'fix_mu2': True}
general_options = {'mu': 1, 'fix_mu': True, 'mu1': 0, 'mu2': 0}
nuis_options = {}

# Degrees of freedom. Free parameters (not including mu) minus nuisance parameters
DOF = 2

# Given a signal hypothesis, create a model we can test for effect size using just
# the free parameter 'mu'
def make_mu_model(s):
    # Create new parameter mapping functions with 'mu1' and 'mu2' parameters fixed.
    # The 'partial' tool from functools is super useful for this.
    s_model = jtm.ParameterModel([jtd.TransDist(sps.norm, partial(pars1, mu1=s[0])),
                                  jtd.TransDist(sps.norm, partial(pars2, mu2=s[1]))]
                                 ,[['mu'],['mu']])
    return s_model 

e.general_model    = general_model
e.make_mu_model    = make_mu_model
e.null_parameters  = null_parameters
e.get_seeds        = get_seeds
e.get_seeds_null   = get_seeds_null
e.null_options     = null_options
e.nuis_options     = nuis_options
e.general_options  = general_options
e.observed_data    = observed_data
e.DOF              = DOF                  

experiments = [e]
