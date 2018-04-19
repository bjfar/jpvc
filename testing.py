"""Some beginning testing stuff"""

import JMCtools as jt
import JMCtools.models as jtm
import JMCtools.distributions as jtd
import JMCtools.common as c

import scipy.stats as sps
import matplotlib.pyplot as plt
import numpy as np
import time

import concurrent.futures

# To import experiments it is nice to do it like this,
# so that we can iterate over them
import importlib
experiment_definitions = ["test","CMS_13TeV_2OSLEP_36invfb"]
experiments = [importlib.import_module("experiments.{0}".format(lib)) 
                 for lib in experiment_definitions]

tag = "asympt"
#Nsamples = int(float(tag))
Nsamples = 0

# Ditch the simulations and just compute asymptotic results
asymptotic_only = True

do_MC = True
if asymptotic_only:
    do_MC = False

# Skip to mu_monster
skip_to_mu_mon = False

# Merge all experiments into one giant experiment
# Will have problems with parameter name collisions. Can avoid them
# manually for now, but can fix in the future
class MonsterExperiment:
    """Class to create a single 'monster' experiment out of
       a list of independent experiments
      
       List of stuff that an experiment needs to define:

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

    def __init__(self,experiments,name="Monster"):
        self.name = name
        self.null_parameters = {}
        self.null_options = {}
        self.general_options = {}
        self.nuis_options = {}
        self.DOF = 0
        obs_data_list = []
        parmodels = []
        self.make_mu_model_fs = []
        self.get_seed_fs      = []
        self.get_seed_null_fs = []
        self.experiment_dims = []
        self.experiment_names = []
        for e in experiments:
            self.null_parameters.update(e.null_parameters)
            self.null_options.update(e.null_options)
            self.nuis_options.update(e.nuis_options)
            self.general_options.update(e.general_options)
            self.DOF += e.DOF
            self.make_mu_model_fs += [e.make_mu_model]
            self.get_seed_fs      += [e.get_seeds]
            self.get_seed_null_fs += [e.get_seeds_null]
            self.experiment_dims += [np.sum(e.general_model.model.dims)]
            self.experiment_names += [e.name]
            parmodels += [e.general_model]
            obs_data_list += [e.observed_data]
        # Join observed data along last axis
        self.observed_data = np.concatenate(
                 [o.reshape(1,1,-1) for o in obs_data_list],axis=-1) 
        # Create new 'general' ParameterModel combining 
        # all submodels of all experiments
        self.general_model = self.create_jointmodel(parmodels)

    def create_jointmodel(self,parmodels):
        """Create a single giant ParameterModel out of a list of
        ParameterModels"""
        all_submodels = []
        all_parfs = []
        all_fargs = []
        all_dims = []
        for m in parmodels:
            all_submodels += m.model.submodels
            all_dims += m.model.dims
            all_parfs += m.parfs
            all_fargs += m.submodel_deps
        new_joint = jtd.JointModel(list(zip(all_submodels,all_dims)))
        return jtm.ParameterModel(new_joint,all_parfs,all_fargs)

    def make_mu_model(self,slist): 
        """Create giant ParameterModel for a signal hypothesis 's'
           Might have to re-think this to more easily handle various
           types of signals, especially e.g. signals for unbinned
           likelihoods."""
        jointmodels = []
        for s,f in zip(slist,self.make_mu_model_fs):
            jointmodels += [f(s)]
        return self.create_jointmodel(jointmodels) 

    def get_seeds(self,samples):
        datalist = c.split_data(self,samples,self.experiment_dims)
        print("samples.shape:",samples.shape)
        print("datalist.shapes:",[d.shape for d in datalist])
        seeds = {}
        for data, seedf in zip(datalist,self.get_seed_fs):
            seeds.update(seedf(data))

    def get_seeds_null(self,samples):
        datalist = c.split_data(self,samples,self.experiment_dims)
        seeds = {}
        for data, seedf in zip(datalist,self.get_seed_null_fs):
            seeds.update(seedf(data))

# Create monster joint experiment
m = MonsterExperiment(experiments)

# Helper plotting function
def makeplot(ax, tobin, theoryf, log=True, label="", c='r', obs=None, pval=None,
             title=None):
    ran = (0,25)
    yran = (1e-3,0.5)
    if tobin is not None:
       n, bins = np.histogram(tobin, bins=50, normed=True, range=ran)
       print("Histogram y range:", np.min(n[n!=0]),np.max(n))
       ax.plot(bins[:-1],n,drawstyle='steps-post',label=label,c=c)
    q = np.arange(ran[0],ran[1],0.01)
    if theoryf is not None:
        ax.plot(q, theoryf(q),c='k')
    ax.set_xlabel("LLR")
    ax.set_ylabel("pdf(LLR)")
    if log:
        #ax.set_ylim(np.min(n[n!=0]),10*np.max(n))
        ax.set_yscale("log")     
    if obs is not None:
        # Draw line for observed value, and show p-value region shaded
        qfill = np.arange(obs,ran[1],0.01)
        if theoryf!=None:
           ax.fill_between(qfill, 0, theoryf(qfill), lw=0, facecolor=c, alpha=0.2)
        pval_str = ""
        if pval!=None:
           pval_str = " (p={0:.2g})".format(pval)
        ax.axvline(x=obs,lw=2,c=c,label="Observed ({0}){1}".format(label,pval_str))
    ax.set_xlim(ran[0],ran[1])
    ax.set_ylim(yran[0],yran[1])
    if title is not None:
        ax.set_title(title)

def fit_general_model(e,samples):
   print("Fitting experiment {0}".format(e.name))

   # Collect data about experiment and how to fit it 
   model = e.general_model

   # Get options for fitting routines
   null_opt = e.null_options
   gen_opt  = e.general_options
  
   if do_MC:
      # Get seeds for fitting routines, tailored to simulated data 
      seeds = e.get_seeds(samples)
      null_seeds = e.get_seeds_null(samples)
     
      # Do some fits!
      Nproc = 3
      print("Fitting null hypothesis...")
      Lmax0, pmax0 = model.find_MLE_parallel(null_opt,samples,method='minuit',
                                             Nprocesses=Nproc,seeds=null_seeds)
      print("Fitting alternate hypothesis (free signal)...")
      Lmax, pmax = model.find_MLE_parallel(gen_opt,samples,method='minuit',
                                           Nprocesses=Nproc,seeds=seeds)

      # Likelihood ratio test statistics
      LLR = -2*(Lmax0 - Lmax)
   else:
      LLR = None

   # Also fit the observed data so we can compute its p-value 
   print("Fitting with observed data...")
   odata = e.observed_data
   Lmax0_obs, pmax0_obs = model.find_MLE_parallel(null_opt, odata, 
                 method='minuit', Nprocesses=1, seeds=e.get_seeds(odata))
   Lmax_obs, pmax_obs = model.find_MLE_parallel(gen_opt, odata, 
                 method='minuit', Nprocesses=1, seeds=e.get_seeds_null(odata))
 
   # Asymptotic p-values
   LLR_obs = -2 * (Lmax0_obs[0] - Lmax_obs[0])
   pval = 1 - sps.chi2.cdf(LLR_obs, e.DOF) 

   return LLR, LLR_obs, pval, e.DOF

def fit_mu_model(e,samples,s):
   """Simulate the 'mu' hypothesis test
   Need to generate the version of the model with the 'mu' parameter for this
   Also need some signal shape to test. For testing purposes we just use a
   test value provided by each experiment."""

   print("Fitting experiment {0}".format(e.name))

   # Create 'mu' version of model with fixed signal shape
   model = e.make_mu_model(s)

   # Get options for fitting routines
   # Only need nuisance options this time
   # Removed fixed status for 'mu' in alternate hypothesis fit
   null_opt = {**e.nuis_options, 'mu': 0, 'fix_mu': True}
   alt_opt = {**e.nuis_options, 'mu': 0.5, 'fix_mu': False}

   if do_MC:
      # Get seeds for fitting routines, tailored to simulated data 
      # Only need the nuisance parameter seeds this time
      null_seeds = e.get_seeds_null(samples)
      
      # Do some fits!
      Nproc = 3
      print("Fitting null hypothesis...")
      Lmax0, pmax0 = model.find_MLE_parallel(null_opt,samples,method='minuit',
                                             Nprocesses=Nproc,seeds=null_seeds)
      print("Fitting alternate hypothesis (free signal)...")
      Lmax, pmax = model.find_MLE_parallel(alt_opt,samples,method='minuit',
                                           Nprocesses=Nproc,seeds=null_seeds)
  
      # Likelihood ratio test statistics
      LLR = -2*(Lmax0 - Lmax)
   else:
      LLR = None

   # Fit the observed data so we can compute its p-value 
   print("Fitting with observed data...")
   odata = e.observed_data
   Lmax0_obs, pmax0_obs = model.find_MLE_parallel(null_opt, odata, 
                 method='minuit', Nprocesses=1, seeds=e.get_seeds_null(odata))
   Lmax_obs, pmax_obs = model.find_MLE_parallel(alt_opt, odata, 
                 method='minuit', Nprocesses=1, seeds=e.get_seeds_null(odata))
 
   # Asymptotic p-values
   LLR_obs = -2 * (Lmax0_obs[0] - Lmax_obs[0])
   pval = 1 - sps.chi2.cdf(LLR_obs, 1) # Only one unprofiled parameter, mu. 

   return LLR, LLR_obs, pval

# Simulate data
all_samples = []
for e in experiments:
   if do_MC:
      all_samples += [e.general_model.simulate(Nsamples,1,e.null_parameters)]
   else:
      all_samples += [[]]

if not skip_to_mu_mon:
   # Main loop for fitting experiments 
   LLR_monster = 0
   LLR_obs_monster = 0
   for j,(e,samples) in enumerate(zip(experiments,all_samples)):
      # Do fit!
      LLR, LLR_obs, pval, e.DOF = fit_general_model(e,samples)
       
      # Save LLR for combining (only works if experiments have no common parameters)
      if LLR is not None:
         LLR_monster += LLR
      else:
         LLR_monster = None
      LLR_obs_monster += LLR_obs
      
      # Plot! 
      fig= plt.figure(figsize=(6,4))
      ax = fig.add_subplot(111)
      makeplot(ax, LLR, lambda q: sps.chi2.pdf(q, e.DOF), log=True, 
              label='free s', c='g', obs=LLR_obs, pval=pval, title=e.name)
      ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
      fig.savefig('auto_experiment_{0}_{1}.png'.format(e.name,tag))
   
      # Fit mu model
      mu_LLR, mu_LLR_obs, mu_pval = fit_mu_model(e,samples,e.s_MLE)
   
      # Plot! 
      fig= plt.figure(figsize=(6,4))
      ax = fig.add_subplot(111)
      makeplot(ax, mu_LLR, lambda q: sps.chi2.pdf(q, 1), log=True, 
              label='mu', c='b', obs=mu_LLR_obs, pval=mu_pval, title=e.name)
      ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
      fig.savefig('auto_experiment_mu_{0}_{1}.png'.format(e.name,tag))
   
   
   # Plot monster LLR distribution
   fig= plt.figure(figsize=(6,4))
   ax = fig.add_subplot(111)
   monster_DOF = np.sum([e.DOF for e in experiments])
   monster_pval = 1 - sps.chi2.cdf(LLR_obs_monster, monster_DOF) 
   makeplot(ax, LLR_monster, lambda q: sps.chi2.pdf(q, monster_DOF), 
            log=True, label='free s', c='g',
            obs=LLR_obs_monster, pval=monster_pval, title="Monster")
   ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
   fig.savefig('auto_experiment_monster_{0}.png'.format(tag))

# Join all samples
if do_MC:
   monster_samples = np.concatenate([samp.reshape(Nsamples,1,-1) 
                                for samp in all_samples],axis=-1)
else:
   monster_samples = None

slist = [e.s_MLE for e in experiments]
mu_LLR, mu_LLR_obs, mu_pval = fit_mu_model(m,monster_samples,slist)

# Plot! 
fig= plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
makeplot(ax, mu_LLR, lambda q: sps.chi2.pdf(q, 1), log=True, 
        label='mu', c='b', obs=mu_LLR_obs, pval=mu_pval, title="Monster")
ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
fig.savefig('auto_experiment_mu_monster_{0}.png'.format(tag))

