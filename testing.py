"""Some beginning testing stuff

We do the following two sorts of statistical tests
'gof' - Goodness of fit. Each experiment characterises itself in terms of a
        number of free parameters, which are fit independently. A single
        'null hypothesis' set of parameters is supplied for testing (e.g.
        the predictions of the best fit point of some scan).
        We simulate under this null hypothesis, and fit the free experimental
        parameters. We then test with L(null)/L(BF_free).

'mu'  - Signal strength test. Predictions of e.g. best fit point are scaled
        universally for all experiments with a signal strength parameter.
        Note that these must be framed as an *extra* contribution to some
        null prediction, e.g. Standard Model. We simulate under the 
        Standard Model null hypothesis that mu=0, and test L(mu=0)/LR(BF_mu).

        NOTE! The null hypothesis is very different in each of these tests!
        So the simulated data is also completely different.
        In the 'gof' test we are trying to exclude e.g. the best fit point
        of a scan.
        In the 'mu' test we are trying to exclude the mu=0 hypothesis, which
        is certainly NOT the best fit point in a scan. However, we do use
        the scan best fit point as input; this provides the *alternate* 
        signal hypothesis which is scaled by 'mu'. So in a sense we are
        testing if mu=0 can be excluded in favour of our scan best fit,
        though it is not quite that simple since we essentially 'cherry-pick'
        this signal hypothesis post-fit, so a certain look-elsewhere effect
        is neglected.

        Note also that the 'mu' test doesn't actually make sense for certain
        observables that constrain the SM and the 'new physics' model in
        essentially the same way. For example take alpha_s; this is a nuisance
        parameter in the fit, constrained to observation, but there is no
        'signal' to be found here. So there is no 'SM' vs 'new_physics' sort
        of test to do with it (because we just fit the SM to observation, it 
        isn't predicted). So it can only contribute to the 'gof' test. It 
        doesn't make sense for the 'mu' test. There is no separable 'new 
        physics' contribution to alpha_s that we could scale with 'mu' here.
"""

import JMCtools as jt
import JMCtools.models as jtm
import JMCtools.distributions as jtd
import JMCtools.common as c

import scipy.stats as sps
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

import concurrent.futures

# To import experiments it is nice to do it like this,
# so that we can iterate over them
import importlib

def run():
    #experiment_definitions = ["CMS_13TeV_2OSLEP_36invfb"] #"gaussian"
    experiment_definitions = ["ColliderBit_analysis"]
    experiment_modules = [importlib.import_module("experiments.{0}".format(lib)) 
                     for lib in experiment_definitions]
    
    # Set the null hypothesis for 'gof' tests. This means fixing the non-nuisance
    # parameters in all of the experiments to something.
    # For testing I am just matching these exactly to observed data: in reality
    # they should come from some model of interested, e.g. a scan best-fit
    gof_null = {}
    gof_null["top_mass"] = {'loc': 173.34}
    gof_null["alpha_s"]  = {'loc': 0.1181}
    gof_null["Z_invisible_width"] = {'loc': 0.2}
    def Collider_Null(N_SR):
       null_s = {"s_{0}".format(i): 0 for i in range(N_SR)} # Actually we'll leave this as zero signal for testing
       #null_theta = {"theta_{0}".format(i): 0 for i in range(7)}
       #null_parameters = {"mu": 0 , **null_s, **null_theta}
       return null_s #parameters
    
    for key, val in experiment_modules[0].experiments.items():
       gof_null[key] = Collider_Null(val.N_SR)
    
    # The 'mu' hypothesis null parameters should be defined internally by each experiment.
    
    # Extract all experiments from modules (some define more than one experiment)
    # Not all experiments are used in all tests, so we also divide them up as needed
    gof_experiments = []
    mu_experiments = [] 
    for em in experiment_modules:
        for e in em.experiments.values():
            if 'gof' in e.tests.keys(): gof_experiments += [e]
            if 'mu'  in e.tests.keys(): mu_experiments  += [e]
            #break
    
    tag = "5e2"
    Nsamples = int(float(tag))
    #Nsamples = 0
    
    # Ditch the simulations and just compute asymptotic results
    # This is very fast! Only have to fit nuisance parameters under
    # the observed data. Could even save those and do them only once
    # ever, but meh.
    if Nsamples == 0:
       asymptotic_only = True
    else:
       asymptotic_only = False
    
    do_MC = True
    if asymptotic_only:
        do_MC = False
    
    # Do single-parameter mu scaling fit?
    do_mu = False 
    
    # Skip to mu_monster
    skip_to_mu_mon = False
    
    # Analyse full combined model?
    do_monster = False
    
    # Dictionary of results
    results = {}
    
    # Actually, the GOF best should work a bit differently. Currently we
    # simulate under the background-only hypothesis, which is fine for
    # testing the goodness of fit of the data to the background-only
    # hypothesis, but we also want to test the goodness-of-fit of e.g.
    # a GAMBIT best fit point. For that, we need to simulate under
    # some best-fit signal hypothesis.
    
    
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
    
        def __init__(self,experiments,name="Monster",common_pars=[]):
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
            # Create new 'general' ParameterModel combining 
            # all submodels of all experiments
            for e in experiments:
                parmodels += [e.general_model]
            self.general_model, renaming = self.create_jointmodel(parmodels,common_pars)
            #print("renaming:",renaming)
     
            # Collect the rest of the data about each experiment, and
            # performing any parameter renaming that was needed to avoid
            # name collisions in the new JointModel.
            for e, rename in zip(experiments,renaming):
                self.null_parameters.update(self.rename_options(e.null_parameters,rename))
                self.null_options.update(self.rename_options(e.null_options,rename))
                self.nuis_options.update(self.rename_options(e.nuis_options,rename))
                self.general_options.update(self.rename_options(e.general_options,rename))
                self.DOF += e.DOF
                self.make_mu_model_fs += [e.make_mu_model]
                self.get_seed_fs      += [e.get_seeds]
                self.get_seed_null_fs += [e.get_seeds_null]
                self.experiment_dims  += [np.sum(e.general_model.model.dims)]
                self.experiment_names += [e.name]
                obs_data_list += [e.observed_data]
            # Join observed data along last axis
            self.observed_data = np.concatenate(
                     [o.reshape(1,1,-1) for o in obs_data_list],axis=-1) 
    
        def rename_options(self,options,renaming):
            # Work out renaming
            rename_dict = {}
            for instruction in renaming:
                new,old = instruction.split(' -> ')
                rename_dict[old] = new
            # Collect and rename options/parameters
            # This is tricky in general since options could have
            # weird names depending on parameters. So for now this
            # only works for the Minuit options.
            # We replace 'par' in the following:
            # 'par'
            # 'error_par'
            # 'fix_par'
            new_opts = {}
            for key,val in options.items():
                newkey = key
                words = key.split("_")
                if len(words)==1:
                    if words[0] in rename_dict.keys():
                        newkey = rename_dict[key]
                if len(words)==2:
                    if words[0] == 'error' or words[0] == 'fix':
                        par = '_'.join(words[1:])
                        if par in rename_dict.keys():
                            newkey = '{0}_{1}'.format(words[0],rename_dict[par])
                new_opts[newkey] = val
            return new_opts 
    
        def create_jointmodel(self,parmodels,common_pars=[]):
            """Create a single giant ParameterModel out of a list of
            ParameterModels"""
            all_submodels = []
            all_fargs = []
            all_dims = []
            all_renaming = []
            for i,m in enumerate(parmodels):
                # Collect submodels and perform parameter renaming to avoid
                # collisions, except where parameters are explicitly set
                # as being common.
                all_renaming += [[]]
                for submodel in m.model.submodels:
                    temp = jtd.TransDist(submodel) # Need this to figure out parameter names
                    renaming = ['Exp{0}_{1} -> {1}'.format(i,par) for par in temp.args if par not in common_pars] 
                    #print(renaming, temp.args, common_pars)
                    all_renaming[i] += renaming  
                    all_submodels += [jtd.TransDist(submodel,renaming_map=renaming)]
                all_dims += m.model.dims
                all_fargs += m.submodel_deps
            new_joint = jtd.JointModel(list(zip(all_submodels,all_dims)))
            return jtm.ParameterModel(new_joint,all_fargs), all_renaming
    
        def make_mu_model(self,slist): 
            """Create giant ParameterModel for a signal hypothesis 's'
               Might have to re-think this to more easily handle various
               types of signals, especially e.g. signals for unbinned
               likelihoods."""
            jointmodels = []
            for s,f in zip(slist,self.make_mu_model_fs):
                jointmodels += [f(s)]
            return self.create_jointmodel(jointmodels,common_pars=['mu'])[0] 
    
        def get_seeds(self,samples):
            datalist = c.split_data(samples,self.experiment_dims)
            print("samples.shape:",samples.shape)
            print("datalist.shapes:",[d.shape for d in datalist])
            seeds = {}
            for data, seedf in zip(datalist,self.get_seed_fs):
                seeds.update(seedf(data))
    
        def get_seeds_null(self,samples):
            datalist = c.split_data(samples,self.experiment_dims)
            seeds = {}
            for data, seedf in zip(datalist,self.get_seed_null_fs):
                seeds.update(seedf(data))
    
    # Create monster joint experiment
    if do_monster:
        m = MonsterExperiment(experiments)
    
    # Helper plotting function
    def makeplot(ax, tobin, theoryf, log=True, label="", c='r', obs=None, pval=None, 
                 qran=None, title=None):
        print("Generating test statistic plot {0}".format(label))
        if qran is None:
            ran = (0,25)
        else:
            ran = qran
        yran = (1e-4,0.5)
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
               print("pval:", pval)
               pval_str = " (p={0:.2g})".format(pval)
            ax.axvline(x=obs,lw=2,c=c,label="Observed ({0}){1}".format(label,pval_str))
        ax.set_xlim(ran[0],ran[1])
        ax.set_ylim(yran[0],yran[1])
        if title is not None:
            ax.set_title(title)
    
    # Simulate data and prepare results dictionaries
    all_samples = []
    for e in gof_experiments:
       print(e.name)
       #print(e.general_model)
       #print(e.general_model.model)
       #print(e.general_model.model.submodels)
       #print(e.general_model.model.submodels[0].submodels)
       print("test_pars:",e.tests['gof'].test_pars)
       if do_MC:
          all_samples += [e.general_model.simulate(Nsamples,e.tests['gof'].test_pars)] # Just using test parameter values
       else:
          all_samples += [[]]
       results[e.name] = {}
    
    LLR_obs_monster = 0
    if not skip_to_mu_mon:
       # Main loop for fitting experiments 
       LLR_monster = 0
       for j,(e,samples) in enumerate(zip(gof_experiments,all_samples)):
          # Do fit!
          test_parameters = gof_null[e.name] # replace this with e.g. prediction from MSSM best fit
          LLR, LLR_obs, pval, epval, gofDOF = e.do_gof_test(test_parameters,samples)
          # Save LLR for combining (only works if experiments have no common parameters)
          if LLR is not None:
             LLR_monster += LLR
          else:
             LLR_monster = None
          LLR_obs_monster += LLR_obs
          
          # Plot! 
          fig= plt.figure(figsize=(6,4))
          ax = fig.add_subplot(111)
          # Range for test statistic axis. Draw as far as is equivalent to 5 sigma
          qran = [0, sps.chi2.ppf(sps.chi2.cdf(25,df=1),df=gofDOF)]  
          makeplot(ax, LLR, lambda q: sps.chi2.pdf(q, gofDOF), log=True, 
                  label='free s', c='g', obs=LLR_obs, pval=pval, qran=qran, 
                   title=e.name+" (Nbins={0})".format(gofDOF))
          ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
          fig.savefig('auto_experiment_{0}_{1}.png'.format(e.name,tag))
          plt.close(fig)

          # Fit mu model
          if do_mu:
              mu_LLR, mu_LLR_obs, mu_pval, mu_epval, muDOF = e.do_mu_test(e.tests['mu'].test_signal,samples)
       
              # Plot! 
              fig= plt.figure(figsize=(6,4))
              ax = fig.add_subplot(111)
              makeplot(ax, mu_LLR, lambda q: sps.chi2.pdf(q, muDOF), log=True, #muDOF should just be 1 
                      label='mu', c='b', obs=mu_LLR_obs, pval=mu_pval, title=e.name)
              ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
              fig.savefig('auto_experiment_mu_{0}_{1}.png'.format(e.name,tag))
              plt.close(fig)

          # Store results
          results[e.name]["LLR_gof_b"]      = LLR_obs
          results[e.name]["apval_gof_b"]    = pval
          results[e.name]["asignif. gof_b"] = -sps.norm.ppf(pval) #/2.) I prefer two-tailed but Andrew says 1-tailed is the convention...
          results[e.name]["DOF"]            = gofDOF 
          if do_mu:
              results[e.name]["LLR_mu_b"]       = mu_LLR_obs
              results[e.name]["apval_mu_b"]     = mu_pval
              results[e.name]["asignif. mu_b"]  = -sps.norm.ppf(mu_pval)
          if LLR is not None:
              results[e.name]["epval_gof_b"]    = epval
              results[e.name]["esignif. gof_b"] = -sps.norm.ppf(epval) #/2.) I prefer two-tailed but Andrew says 1-tailed is the convention...
              if do_mu:
                 results[e.name]["epval_mu_b"]     = mu_epval
                 results[e.name]["esignif. mu_b"]  = -sps.norm.ppf(mu_epval)
     
       a = np.argsort(LLR)
       #print("LLR_monster:",LLR_monster[a])
     
       # Plot monster LLR distribution
       fig= plt.figure(figsize=(6,4))
       ax = fig.add_subplot(111)
       monster_DOF = np.sum([e.DOF for e in gof_experiments])
       monster_pval = 1 - sps.chi2.cdf(LLR_obs_monster, monster_DOF)
       monster_epval = c.e_pval(LLR_monster,LLR_obs_monster) if do_MC else None
       makeplot(ax, LLR_monster, lambda q: sps.chi2.pdf(q, monster_DOF), 
                log=True, label='free s', c='g',
                obs=LLR_obs_monster, pval=monster_pval, title="Monster")
       ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
       fig.savefig('auto_experiment_monster_{0}.png'.format(tag))
       plt.close(fig)

    # Join all samples
    if do_MC:
       monster_samples = np.concatenate([samp.reshape(Nsamples,1,-1) 
                                    for samp in all_samples],axis=-1)
    else:
       monster_samples = None
    
    if do_mu and do_monster:
       slist = [e.s_MLE for e in experiments]
       mu_LLR, mu_LLR_obs, mu_pval, mu_epval = fit_mu_model(m,monster_samples,slist)
       
       # Plot! 
       fig= plt.figure(figsize=(6,4))
       ax = fig.add_subplot(111)
       makeplot(ax, mu_LLR, lambda q: sps.chi2.pdf(q, 1), log=True, 
               label='mu', c='b', obs=mu_LLR_obs, pval=mu_pval, title="Monster")
       ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
       fig.savefig('auto_experiment_mu_monster_{0}.png'.format(tag))
       plt.close(fig)

    # Store results for Monster
    results["Combined"] = {}
    results["Combined"]["LLR_gof_b"]   = LLR_obs_monster
    results["Combined"]["apval_gof_b"]  = monster_pval
    results["Combined"]["asignif. gof_b"] = -sps.norm.ppf(monster_pval)
    results["Combined"]["DOF"]         = monster_DOF 
    if do_mu and do_monster:
       results["Combined"]["LLR_mu_b"]    = mu_LLR_obs
       results["Combined"]["apval_mu_b"]   = mu_pval
       results["Combined"]["asignif. mu_b"]  = -sps.norm.ppf(mu_pval)
    if do_MC:
       results["Combined"]["epval_gof_b"]  = monster_epval
       results["Combined"]["esignif. gof_b"] = -sps.norm.ppf(monster_epval)
    if do_MC and do_mu and do_monster:
       results["Combined"]["epval_mu_b"]   = mu_epval
       results["Combined"]["esignif. mu_b"]  = -sps.norm.ppf(mu_epval)
    
    # Ok let's produce some nice tables of results. Maybe even
    # some cool bar graphs showing the "pull" of each experiment
    
    # Convert results to Pandas dataframe
    r = pd.DataFrame.from_dict(results)
    order = ['DOF',
              'LLR_gof_b',
              'apval_gof_b']
    if do_MC: order += ['epval_gof_b']
    order += ['asignif. gof_b']
    if do_MC: order += ['esignif. gof_b']
    if do_mu: order += ['LLR_mu_b', 
                        'apval_mu_b'] 
    if do_MC and do_mu: order += ['epval_mu_b']
    if do_mu: order += ['asignif. mu_b']
    if do_MC and do_mu: order += ['esignif. mu_b']
    exp_order = [e.name for e in gof_experiments] + ['Combined']
    print(r[exp_order].reindex(order))

if __name__ == "__main__":
    run() 
