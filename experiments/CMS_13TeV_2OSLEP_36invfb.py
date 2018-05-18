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
import matplotlib.pyplot as plt
from .experiment import Experiment

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

# Nominal signal parameters, for testing. User should provide this from their model 
test_signal = {'s_{0}'.format(i): s_MLE[i] for i in range(N_regions)}

# Function to map signal hypothesis into Poisson distribution parameters
def poisson_f(i, s, theta):
    l = np.atleast_1d(s + CMS_b[i] + theta)
    m = (l<0)
    l[m] = 0  #Poisson cannot have negative mean
    return {'mu': l}

# Parameter mapping function for nuisance parameter constraints
def func_nuis(**thetas):
    #print("in func_nuis:", thetas)
    means = np.array([thetas['theta_{0}'.format(i)] for i in range(N_regions)])
    return {'mean': means.flatten(),
             'cov': CMS_cov}

# Create the transformed pdf functions
# Also requires some parameter renaming since we use the
# same underlying function repeatedly
poisson_part = [jtd.TransDist(sps.poisson,partial(poisson_f,i),
                       ['s_{0} -> s'.format(i), 
                        'theta_{0} -> theta'.format(i)])
                 for i in range(N_regions)]
corr_dist = jtd.TransDist(sps.multivariate_normal,func_nuis,
               func_args=["theta_{0}".format(i) for i in range(N_regions)])
correlations = [(corr_dist,7)]

# Create the joint PDF object
joint     = jtd.JointDist(poisson_part + correlations)
 
# Set options for parameter fitting
theta_opt  = {'theta_{0}'.format(i) : 0 for i in range(N_regions)}
theta_opt2 = {'error_theta_{0}'.format(i) : 1.*np.sqrt(CMS_cov[i][i]) for i in range(N_regions)} # Get good step sizes from covariance matrix
s_opt  = {'s_{0}'.format(i): 0 for i in range(N_regions)} # Maybe zero is a good starting guess? Should use seeds that guess based on data.
s_opt2 = {'error_s_{0}'.format(i) :  0.1*np.sqrt(CMS_cov[i][i]) for i in range(N_regions)} # Get good step sizes from covariance matrix.
s_options = {**s_opt, **s_opt2}

nuis_options = {**theta_opt, **theta_opt2}
general_options = {**s_options, **nuis_options}

# Functions to provide starting guesses for parameters, tuned to each MC sample realisation
def get_seeds_full(samples):
   """Gets seeds for s and theta fits"""
   seeds={}
   bin_samples = samples[:,0,:N_regions].T
   theta_samples = samples[:,0,N_regions:].T
   for i in range(N_regions):
      theta_MLE = theta_samples[i]
      s_MLE = bin_samples[i] - CMS_b[i] - theta_MLE
      seeds['theta_{0}'.format(i)] = theta_MLE
      seeds['s_{0}'.format(i)] = s_MLE 
      #print('seeds for s_{0}: {1}'.format(i,s_MLE))
   return seeds

def get_seeds_null(samples):
   """Gets seeds for just nuisance parameters fits"""
   theta_seeds={}
   theta_samples = samples[:,0,N_regions:].T
   for i in range(N_regions):
      theta_MLE = theta_samples[i]
      theta_seeds['theta_{0}'.format(i)] = theta_MLE
   return theta_seeds

# Full observed data list, included observed values of nuisance measurements
observed_data = np.concatenate([np.array(CMS_o),np.zeros(N_regions)],axis=-1)

# Can define extra calculations to be done or plots to be created using the fit
# results, to help diagnose any problems with the fits. 
def dfull(e, Lmax0, pmax0, Lmax, pmax):
    # Plot distribution of fit values against their
    # true values under the null hypothesis. Make sure
    # this makes sense.
 
    expected = {**s_opt,**theta_opt}

    fig = plt.figure(figsize=(14,6))
    N = len(pmax.keys())
    for i in range(14):
        if i % 2 == 0:
           key = 's_{0}'.format(i//2)
           pos = i//2 + 1
        elif i % 2 == 1:
           key = 'theta_{0}'.format(i//2)
           pos = i//2 + 1 + 7
        val = pmax[key] 
        val = val[np.isfinite(val)] # remove nans from failed fits
        n, bins = np.histogram(val, normed=True)
        ax = fig.add_subplot(2,7,pos)
        ax.plot(bins[:-1],n,drawstyle='steps-post',label="")
        ax.set_title(key)
        trueval = expected[key]
        ax.axvline(trueval,lw=2,c='k')
    plt.tight_layout()
    fig.savefig('{0}_diagnostic_full.png'.format(e.name))
    
def dnull(e, Lmax0, pmax0, Lmax, pmax):
    # Plot distribution of fit values against their
    # true values under the null hypothesis. Make sure
    # this makes sense.
 
    expected = {**theta_opt}

    fig = plt.figure(figsize=(14,3))
    N = len(pmax.keys())
    for i in range(7):
        key = 'theta_{0}'.format(i)
        pos = i+1
        val = pmax0[key]
        val = val[np.isfinite(val)] # remove nans from failed fits 
        #print(key, val)
        n, bins = np.histogram(val, normed=True)
        ax = fig.add_subplot(1,7,pos)
        ax.plot(bins[:-1],n,drawstyle='steps-post',label="")
        ax.set_title(key)
        trueval = expected[key]
        ax.axvline(trueval,lw=2,c='k')
    plt.tight_layout()
    fig.savefig('{0}_diagnostic_null.png'.format(e.name))
 
# Define the experiment object and options for fitting during statistical tests
e = Experiment(name,joint,observed_data,DOF=7)
 
e.define_gof_test(nuisance_par_null=theta_opt,
                  test_pars={**s_opt,**theta_opt}, # Just for testing purposes
                  null_options=nuis_options,
                  full_options=general_options,
                  null_seeds=get_seeds_null,
                  full_seeds=get_seeds_full,
                  diagnostics=[dfull,dnull]
                  )

e.define_mu_test(nuisance_par_null=theta_opt,
                 null_options=nuis_options,
                 null_seeds=get_seeds_null,
                 scale_with_mu=['s_{0}'.format(i) for i in range(N_regions)],
                 test_signal=test_signal
                 )

# Return the number of non-nuisance parameters for the fit, for
# degrees-of-freedom calculations
# This applies to the 'general' model, not the 'mu' model.
#e.DOF = 7

experiments = [e]
