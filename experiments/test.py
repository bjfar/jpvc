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

name = "test"

# Create the joint PDF object
joint = jtd.JointModel([sps.norm,sps.norm])

# Create the "observed" data
# Need extra axes for matching shape of many simulated datasets
observed_data = np.array([7,8])[np.newaxis,np.newaxis,:]

# "Background" predictions
b = [5,6]

# s_MLE is not required, just useful for testing
s_MLE = [2,2]

# Create parameter mappings
# Proxy for 'signal strength' parameter added
def pars1(mu,mu1):
    return {"loc": b[0] + mu*mu1, "scale":1}

def pars2(mu,mu2):
    return {"loc": b[1] + mu*mu2, "scale":1}

general_model = jtm.ParameterModel(joint,[pars1,pars2])

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
    funcs = [partial(pars1, mu1=s[0]),
             partial(pars2, mu2=s[1])]
    # However, we then need to explicitly tell the ParameterModel what the function
    # arguments are
    fargs = [['mu'],['mu']]
    return jtm.ParameterModel(joint,funcs,fargs) 

