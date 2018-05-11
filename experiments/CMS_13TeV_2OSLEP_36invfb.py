"""Analysis described in https://arxiv.org/abs/1709.08908,
   i.e. the CMS search including two opposite-sign, same-flavour 
   leptons, jets, and MET at 13 TeV.

   Simplified likelihood construction follows 
   https://cds.cern.ch/record/2242860
"""

import JMCtools.distributions as jtd
import JMCtools.models as jtm
import numpy as np
import scipy.stats as sps
from functools import partial

name = "CMS_13TeV_2OSLEP_36invfb" 

# Observed event counts
CMS_o = [57., 29., 2., 0., 9., 5., 1.]

# Background estimates (means)
CMS_b = [54.9, 21.6, 6., 2.5, 7.6, 5.6, 1.3]
   
# Covariance matrix for nuisance parameter measurements
CMS_cov = [
  [ 52.8, 12.7,  3.0,  1.2,  4.5,  5.1,  1.2 ],
  [ 12.7, 41.4,  3.6,  2.0,  2.5,  2.0,  0.7 ],
  [  3.0,  3.6,  1.6,  0.6,  0.4,  0.3,  0.1 ],
  [  1.2,  2.0,  0.6,  1.1,  0.3,  0.1,  0.1 ],
  [  4.5,  2.5,  0.4,  0.3,  6.5,  1.8,  0.4 ],
  [  5.1,  2.0,  0.3,  0.1,  1.8,  2.4,  0.4 ],
  [  1.2,  0.7,  0.1,  0.1,  0.4,  0.4,  0.2 ],
]

# Maximum likelihood estimators for 's' parameters
# under the observed data, ignoring correlations
# between nuisance observations. Useful for seeding
# certain fits.
s_MLE = np.array(CMS_o) - np.array(CMS_b)

# For this CMS analysis, a simplified joint PDF can be 
# constructed as a product of numerous independent Poisson 
# distributions, which model events in various signal regions, and
# a large multivariate normal distribution which encodes correlations
# between constraints on nuisance parameters for the Poisson
# distributions. So our first task is to build this joint PDF.

N_regions = len(CMS_o)

# The parameter mapping functions are all almost identical
# so we can create them with a generator function
# As a subtlety, we need to add the generated functions to
# the global dict in order to make them picklable, which
# is needed to run them via multiprocessing, which we need
# for our parallelised fitting routines.
def poisson_f(i, mu, s, theta):
    l = np.atleast_1d(mu*s + CMS_b[i] + theta)
    m = (l<0)
    l[m] = 0  #Poisson cannot have negative mean
    return {'mu': l}

# This is some slightly dark voodoo that needs to be
# done so that poisson_f can be pickled, and thus
# used in multiprocessing applications like our
# fitting routines. See
# https://stackoverflow.com/a/49900573/1447953
def make_poisson_func(i):
    return partial(poisson_f,i)
# The partial-ised function will have the 'i' argument
# removed!

# We now have to manually specify the names of
# our parameters, to be used by the JointModel object to infer the
# dependency structure of our model.
def poisson_arg_names(i):
    return ('mu', 's_{0}'.format(i), 'theta_{0}'.format(i))

poisson_fs   = [make_poisson_func(i) for i in range(N_regions)]
poisson_args = [poisson_arg_names(i) for i in range(N_regions)]

# Parameter mapping function for nuisance parameter constraints
def func_nuis(**thetas):
    means = np.array([thetas['theta_{0}'.format(i)] for i in range(N_regions)])
    return {'mean': means.flatten(),
             'cov': CMS_cov}

nuis_fs   = [func_nuis] 
nuis_args = [tuple("theta_{0}".format(i) for i in range(N_regions))]

# Versions which ignore correlations, for cross-checking and to
# help with starting guesses for full model
def nuis_f(i,theta):
    return {'loc': theta,
            'scale': np.sqrt(CMS_cov[i][i])}

def make_nuis_func(i):
    return partial(nuis_f,i)

nuis_fs_noc = [make_nuis_func(i) for i in range(N_regions)]
# Note different placement of tuple parentheses!
# Also note the comma to force it to be a tuple
nuis_args_noc = [("theta_{0}".format(i),) for i in range(N_regions)]

# Create the transformed pdf functions
# Also requires some parameter renaming since we use the
# same underlying function repeatedly
poisson_part = [jtd.TransDist(sps.poisson,make_poisson_func(i),
                       ['s_{0} -> s'.format(i), 
                        'theta_{0} -> theta'.format(i)])
                 for i in range(N_regions)]
corr_dist = jtd.TransDist(sps.multivariate_normal,func_nuis)
correlations = [(corr_dist,7)]

# Let's also create a version which ignores the correlations
# This will be much faster to fit, and should still resemble
# the right answer, so can be a helpful cross-check.
# Just take the diagonal of the multivariate normal.
no_correlations = [jtd.TransDist(sps.norm,make_nuis_func(i)) for i in range(N_regions)]

# Create the joint PDF objects
joint     = jtd.JointModel(poisson_part + correlations)
joint_noc = jtd.JointModel(poisson_part + no_correlations)

# Connect the joint PDFs to the parameter structures
# Set 'mu' parameter to be considered as fixed, since we
# don't plan to use it in these models.
general_model = jtm.ParameterModel(joint, 
                               poisson_args + nuis_args,
                               fix={'mu':1})

# Not using this atm.
model_noc = jtm.ParameterModel(joint_noc, 
                               poisson_args + nuis_args_noc,
                               fix={'mu':1})

# Check the inferred block structures
#print("model.blocks    :", model.blocks)
#print("model_noc.blocks:", model_noc.blocks)

# Check the arguments listed:
#print("submodel args:")
#for a in poisson_args + nuis_args:
#   print("   ",a)

# IMPORTANT!
# To allow automated processing of this experiment, we need to define
# the model to be 'exported' with a special name. This is 'model'.
#
# But we probably want to export some other stuff too. To actually run
# the fits we will need to know what parameters need to be predicted. I
# guess we even want to re-create a joint ParameterModel with new
# function definitions depending on what we want to do. Hmm. Probably
# need to pre-define some standard tasks, e.g. fit 'mu' given nominal
# signal from some GAMBIT best fit VS fit all signal bins of all models
# independently. Maybe we'll just try those two first. Each experiment
# should then probably have some function for creating its own
# ParameterModel given some signal to be tested, e.g.

# Provide decent default options for the fitting as well:
theta_opt  = {'theta_{0}'.format(i) : 0 for i in range(N_regions)}
theta_opt2 = {'error_theta_{0}'.format(i) : 1.*np.sqrt(CMS_cov[i][i]) for i in range(N_regions)} # Get good step sizes from covariance matrix
s_opt  = {'s_{0}'.format(i): 0 for i in range(N_regions)} # Maybe zero is a good starting guess? Should use seeds that guess based on data.
s_opt2 = {'error_s_{0}'.format(i) :  0.1*np.sqrt(CMS_cov[i][i]) for i in range(N_regions)} # Get good step sizes from covariance matrix.
s_options = {**s_opt, **s_opt2}
nuis_options = {**theta_opt, **theta_opt2}
general_options = {'mu': 1, 'fix_mu': True, **s_options, **nuis_options}

fix_s = {'fix_s_{0}'.format(i): True for i in range(N_regions)}
# Options for fitting null hypothesis
null_options = {'mu': 0, 'fix_mu': True, **s_opt, **fix_s, **nuis_options}

# If doing multiple fits, need to provide unique tags so
# that parameter mapping functions can be added to the
# 'globals' dict without collision.
def mu_f(i,s,mu,theta):
    return poisson_fs[i](mu,s,theta)

def make_mu_func(i,s):
    return partial(mu_f,i,s)

def make_mu_model(s):
    mu_funcs = []
    mu_args = []
    mu_argmap = []
    for i in range(N_regions):
        # mu and theta will still have to be profiled out
        mu_funcs += [make_mu_func(i,s[i])]
        mu_args  += [('mu','theta_{0}'.format(i))]       
        mu_argmap+= [['theta_{0} -> theta'.format(i)]]

    poisson_mu_part = [jtd.TransDist(sps.poisson,mu_funcs[i],mu_argmap[i]) for i in range(N_regions)]
    joint = jtd.JointModel(poisson_mu_part + correlations)
    model = jtm.ParameterModel(joint, mu_args + nuis_args)

    # Will need to set options for 'mu' and 'error_mu'
    # externally. This will depend on the test statistic.
    # Nuisance parameter options in 'nuisance_options' can be used.
    return model

# We should probably also provide some pre-set 'good' parameters for
# doing the fitting. Or even better, a function that can compute
# these from the data. Maybe that should be something that can be added
# to the ParameterModel?

def get_seeds(samples):
   """Gets seeds for s and theta fits"""
   seeds={}
   bin_samples = samples[:,0,:N_regions].T
   theta_samples = samples[:,0,N_regions:].T
   for i in range(N_regions):
      theta_MLE = theta_samples[i]
      s_MLE = bin_samples[i] - CMS_b[i] - theta_MLE
      seeds['theta_{0}'.format(i)] = theta_MLE
      seeds['s_{0}'.format(i)] = s_MLE 
   return seeds

def get_seeds_null(samples):
   """Gets seeds for just nuisance parameters fits"""
   theta_seeds={}
   theta_samples = samples[:,0,N_regions:].T
   for i in range(N_regions):
      theta_MLE = theta_samples[i]
      theta_seeds['theta_{0}'.format(i)] = theta_MLE
   return theta_seeds

# Return the number of non-nuisance parameters for the fit, for
# degrees-of-freedom calculations
# This applies to the 'general' model, not the 'mu' model.
DOF = 7

# Define parameters for simulating background-only hypothesis in general_model
null_s = {"s_{0}".format(i): 0 for i in range(N_regions)}
null_theta = {"theta_{0}".format(i): 0 for i in range(N_regions)}
null_parameters = {"mu": 0 , **null_s, **null_theta}

# Compute observed maximum likelihood values
observed_data = np.concatenate([np.array(CMS_o),np.zeros(len(CMS_o))],axis=-1)
# Need to add the Ntrials and Ndraws axes
observed_data = observed_data[np.newaxis,np.newaxis,:]
seed_obs = {'s_{0}'.format(i): [s_MLE[i]] for i in range(N_regions)} # change seeds to fit observed data 

#print(observed_data)
#print(get_seeds(observed_data))
#print(null_options)
#print("Getting observed MLEs")
#null_options['print_level'] = 1 # be noisy for testing

# Huh, seems like this hangs if we do it during the module import...
#Lmax0_obs, pmax0_obs = general_model.find_MLE_parallel(null_options,observed_data,method='minuit',Nprocesses=1,seeds=get_seeds(observed_data))
#Lmax_obs, pmax_obs   = general_model.find_MLE_parallel(general_options,observed_data,method='minuit',Nprocesses=1,seeds=get_seeds_null(observed_data))
#
## Asymptotic p-values for general fit
#general_LLR_obs = -2 * (Lmax0_obs[0] - Lmax2_obs[0])
#genral_pval = 1 - sps.chi2.cdf(general_LLR_obs, DOF) 

# Current list of required objects:
# name
# general_model
# make_mu_model
# null_parameters
# get_seeds
# get_seeds_null
# null_options
# general_options
# observed_data
# DOF




