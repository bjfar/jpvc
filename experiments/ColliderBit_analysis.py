"""Tools for setting up an experiment constructed from a GAMBIT
   ColliderBit analysis.
   We get information about these analyses from GAMBIT in a
   standardised format, so it is possible to mostly automate
   the construction of their pdfs.
"""

import JMCtools.distributions as jtd
import JMCtools.models as jtm
import numpy as np
import scipy.stats as sps
from functools import partial
import matplotlib.pyplot as plt
from .experiment import Experiment

# Function to map signal hypothesis into Poisson distribution parameters
# Signal+background systematics dealt with via a MULTIPLICATIVE factor
def poisson_f_mult(s, b, theta):
    l = theta*np.atleast_1d(s + b)
    m = (l<0)
    l[m] = 0  #Poisson cannot have negative mean
    return {'mu': l}

# Function to map signal hypothesis into Poisson distribution parameters
# Signal+background systematics dealt with via an ADDITIVE factor
def poisson_f_add(s, b, theta):
    l = np.atleast_1d(s + b + theta)
    m = (l<0)
    l[m] = 0  #Poisson cannot have negative mean
    return {'mu': l}

# Parameter mapping function for nuisance parameter constraints
def func_nuis_corr(cov, **thetas):
    #print("in func_nuis:", thetas)
    means = np.array([thetas['theta_{0}'.format(i)] for i in range(len(thetas))])
    return {'mean': means.flatten(),
             'cov': cov}

# Gaussian constraint for additive nuisance parameter
def func_nuis_norm_add(theta,theta_std):
    return {'loc': theta,
             'scale': theta_std}

# Log-normal constraint for multiplicative nuisance parameter
def func_nuis_lognorm_mult(theta,theta_std):
    return {'scale': theta,
            's': theta_std}


class Analysis:
    def __init__(self,name):
        self.name = name
        #self.SR_names
        #self.SR_n    
        #self.SR_b    
        #self.SR_b_sys
        #self.SR_s_sys
        #self.SR_s    
        self.cov = None

    def make_experiment(self):
        """Turn this analysis information into a jpvc experiment"""
        self.N_SR = len(self.SR_names)

        # Maximum likelihood estimators for 's' parameters
        # under the observed data, ignoring correlations
        # between nuisance observations. Useful for seeding
        # certain fits.
        self.s_MLE = np.array(self.SR_n) - np.array(self.SR_b)

        # Nominal signal parameters, for testing. User should provide this from their model 
        self.test_signal = {'s_{0}'.format(i): self.s_MLE[i] for i in range(self.N_SR)}

        if self.cov is not None:
           e = self.make_experiment_cov()
        else:
           e = self.make_experiment_nocov()
        return e        

    # Functions to provide starting guesses for parameters, tuned to each MC sample realisation
    def seeds_full_f_add(self):
        def get_seeds_full(samples):
           """Gets seeds for s and theta fits (additive nuisance)"""
           seeds={}
           bin_samples = samples[:,0,:self.N_SR].T
           theta_samples = samples[:,0,self.N_SR:].T
           for i in range(self.N_SR):
              theta_MLE = theta_samples[i]
              s_MLE = bin_samples[i] - theta_MLE - self.SR_b[i]
              seeds['theta_{0}'.format(i)] = theta_MLE
              seeds['s_{0}'.format(i)] = s_MLE 
              #print('seeds for s_{0}: {1}'.format(i,s_MLE))
           return seeds
        return get_seeds_full

    def seeds_full_f_mult(self):
        def get_seeds_full(samples):
           """Gets seeds for s and theta fits (multiplicative nuisance)"""
           seeds={}
           bin_samples = samples[:,0,:self.N_SR].T
           theta_samples = samples[:,0,self.N_SR:].T
           for i in range(self.N_SR):
              theta_MLE = theta_samples[i]
              s_MLE = bin_samples[i]/theta_MLE - self.SR_b[i]
              seeds['theta_{0}'.format(i)] = theta_MLE
              seeds['s_{0}'.format(i)] = s_MLE 
              #print('seeds for s_{0}: {1}'.format(i,s_MLE))
           return seeds
        return get_seeds_full
    
    def seeds_null_f(self): 
        def get_seeds_null(samples):
           """Gets seeds for (both additive and multiplicative) nuisance parameters fits"""
           theta_seeds={}
           theta_samples = samples[:,0,self.N_SR:].T
           for i in range(self.N_SR):
              theta_MLE = theta_samples[i]
              theta_seeds['theta_{0}'.format(i)] = theta_MLE
           return theta_seeds
        return get_seeds_null

    def make_experiment_cov(self):
        # Create the transformed pdf functions
        # Also requires some parameter renaming since we use the
        # same underlying function repeatedly
        poisson_part = [jtd.TransDist(sps.poisson,partial(poisson_f_add,b=self.SR_b[i]),
                               ['s_{0} -> s'.format(i), 
                                'theta_{0} -> theta'.format(i)])
                         for i in range(self.N_SR)]
        corr_dist = jtd.TransDist(sps.multivariate_normal,partial(func_nuis_corr,cov=self.cov),
                       func_args=["theta_{0}".format(i) for i in range(self.N_SR)])
        correlations = [(corr_dist,self.N_SR)]

        # Create the joint PDF object
        joint = jtd.JointDist(poisson_part + correlations)
         
        # Set options for parameter fitting
        theta_opt  = {'theta_{0}'.format(i) : 0 for i in range(self.N_SR)}
        theta_opt2 = {'error_theta_{0}'.format(i) : 1.*np.sqrt(self.cov[i][i]) for i in range(self.N_SR)} # Get good step sizes from covariance matrix
        s_opt  = {'s_{0}'.format(i): 0 for i in range(self.N_SR)} # Maybe zero is a good starting guess? Should use seeds that guess based on data.
        s_opt2 = {'error_s_{0}'.format(i) :  0.1*np.sqrt(self.cov[i][i]) for i in range(self.N_SR)} # Get good step sizes from covariance matrix.
        s_options = {**s_opt, **s_opt2}
        
        nuis_options = {**theta_opt, **theta_opt2}
        general_options = {**s_options, **nuis_options}

        # Full observed data list, included observed values of nuisance measurements
        observed_data = np.concatenate([np.array(self.SR_n),np.zeros(self.N_SR)],axis=-1)

        # Define the experiment object and options for fitting during statistical tests
        e = Experiment(self.name,joint,observed_data,DOF=self.N_SR)
         
        e.define_gof_test(nuisance_par_null=theta_opt,
                          test_pars={**s_opt,**theta_opt}, # Just for testing purposes
                          null_options=nuis_options,
                          full_options=general_options,
                          null_seeds=self.seeds_null_f(),
                          full_seeds=self.seeds_full_f_add(),
                          diagnostics=[self.make_dfull(s_opt,theta_opt),
                                       self.make_dnull(theta_opt)]
                          )
        
        e.define_mu_test(nuisance_par_null=theta_opt,
                         null_options=nuis_options,
                         null_seeds=(self.seeds_null_f(), True),
                         scale_with_mu=['s_{0}'.format(i) for i in range(self.N_SR)],
                         test_signal=self.test_signal
                         )
        return e


    def make_experiment_nocov(self):
        # Create the transformed pdf functions
        # Also requires some parameter renaming since we use the
        # same underlying function repeatedly
        poisson_part_mult = [jtd.TransDist(sps.poisson,partial(poisson_f_mult,b=self.SR_b[i]),
                               ['s_{0} -> s'.format(i), 
                                'theta_{0} -> theta'.format(i)])
                         for i in range(self.N_SR)]

        poisson_part_add = [jtd.TransDist(sps.poisson,partial(poisson_f_add,b=self.SR_b[i]),
                               ['s_{0} -> s'.format(i), 
                                'theta_{0} -> theta'.format(i)])
                         for i in range(self.N_SR)]

        # Using lognormal constraint on multiplicative systematic parameter
        sys_dist_mult = [jtd.TransDist(sps.lognorm,
                                  partial(func_nuis_lognorm_mult,
                                          theta_std=self.SR_b_sys[i]/self.SR_b[i]),
                                  ['theta_{0} -> theta'.format(i)])
                      for i in range(self.N_SR)]

        # Using normal constaint on additive systematic parameter
        sys_dist_add = [jtd.TransDist(sps.norm,
                                  partial(func_nuis_norm_add,
                                          theta_std=self.SR_b_sys[i]),
                                  ['theta_{0} -> theta'.format(i)])
                      for i in range(self.N_SR)]


        #print("fractional systematic uncertainties:")
        #print([self.SR_b_sys[i]/self.SR_b[i] for i in range(self.N_SR)])
        #quit()

        # Create the joint PDF object
        #joint = jtd.JointDist(poisson_part_mult + sys_dist_mult)
        joint = jtd.JointDist(poisson_part_add + sys_dist_add) 
 
        # Set options for parameter fitting
        #theta_opt  = {'theta_{0}'.format(i) : 1 for i in range(self.N_SR)} # multiplicative
        theta_opt  = {'theta_{0}'.format(i) : 0 for i in range(self.N_SR)} # additive
        theta_opt2 = {'error_theta_{0}'.format(i) : 1.*self.SR_b_sys[i] for i in range(self.N_SR)} # Get good step sizes from systematic error estimate
        s_opt  = {'s_{0}'.format(i): 0 for i in range(self.N_SR)} # Maybe zero is a good starting guess? Should use seeds that guess based on data.
        s_opt2 = {'error_s_{0}'.format(i) :  0.1*self.SR_b_sys[i] for i in range(self.N_SR)} # Get good step sizes from systematic error estimate
        s_options = {**s_opt, **s_opt2}
       
        nuis_options = {**theta_opt, **theta_opt2}
        general_options = {**s_options, **nuis_options}

        # Full observed data list, included observed values of nuisance measurements
        observed_data = np.concatenate([np.array(self.SR_n),np.ones(self.N_SR)],axis=-1)

        # print("Setup for experiment {0}".format(self.name))
        # #print("general_options:", general_options)
        # #print("s_MLE:", self.s_MLE)
        # #print("N_SR:", self.N_SR)
        # #print("observed_data:", observed_data.shape)
        # oseed = self.seeds_full_f_mult()(np.array(observed_data)[np.newaxis,np.newaxis,:])
        # print("parameter, MLE, data, seed")
        # for i in range(self.N_SR):
        #     par = "s_{0}".format(i)
        #     print("{0}, {1}, {2}, {3}".format(par, self.s_MLE[i], observed_data[i], oseed[par]))
        # for i in range(self.N_SR):
        #     par = "theta_{0}".format(i)
        #     print("{0}, {1}, {2}, {3}".format(par, 1, observed_data[i+self.N_SR], oseed[par]))
        # quit()

        # Define the experiment object and options for fitting during statistical tests
        e = Experiment(self.name,joint,observed_data,DOF=self.N_SR)
         
        e.define_gof_test(nuisance_par_null=theta_opt,
                          test_pars={**s_opt,**theta_opt}, # Just for testing purposes
                          null_options=nuis_options,
                          full_options=general_options,
                          null_seeds=(self.seeds_null_f(), True), # Extra flag indicates that the "seeds" are actually the analytically exact MLEs, so no numerical minimisation needed
                          full_seeds=(self.seeds_full_f_add(), True),
                          diagnostics=[self.make_dfull(s_opt,theta_opt),
                                       self.make_dnull(theta_opt)]
                          )
        
        e.define_mu_test(nuisance_par_null=theta_opt,
                         null_options=nuis_options,
                         null_seeds=self.seeds_null_f(),
                         scale_with_mu=['s_{0}'.format(i) for i in range(self.N_SR)],
                         test_signal=self.test_signal
                         )
        return e

    def make_dfull(self,s_opt,theta_opt):
        # Can define extra calculations to be done or plots to be created using the fit
        # results, to help diagnose any problems with the fits. 
        def dfull(e, Lmax0, pmax0, Lmax, pmax):
            # Plot distribution of fit values against their
            # true values under the null hypothesis. Make sure
            # this makes sense.
         
            expected = {**s_opt,**theta_opt}
        
            fig = plt.figure(figsize=(2*self.N_SR,6))
            N = len(pmax.keys())
            for i in range(2*self.N_SR):
                if i % 2 == 0:
                   key = 's_{0}'.format(i//2)
                   pos = i//2 + 1
                elif i % 2 == 1:
                   key = 'theta_{0}'.format(i//2)
                   pos = i//2 + 1 + self.N_SR
                val = pmax[key] 
                val = val[np.isfinite(val)] # remove nans from failed fits
                #val = val[val>=0] # remove non-positive MLEs, these can't be log'd
                n, bins = np.histogram(val, normed=True)
                ax = fig.add_subplot(2,self.N_SR,pos)
                ax.plot(bins[:-1],n,drawstyle='steps-post',label="")
                ax.set_title(key)
                trueval = expected[key]
                ax.axvline(trueval,lw=2,c='k')
            plt.tight_layout()
            fig.savefig('{0}_diagnostic_full.png'.format(e.name))
            plt.close(fig)
        return dfull            

    def make_dnull(self,theta_opt):
        def dnull(e, Lmax0, pmax0, Lmax, pmax):
            # Plot distribution of fit values against their
            # true values under the null hypothesis. Make sure
            # this makes sense.
         
            expected = {**theta_opt}
        
            fig = plt.figure(figsize=(2*self.N_SR,3))
            N = len(pmax.keys())
            for i in range(self.N_SR):
                key = 'theta_{0}'.format(i)
                pos = i+1
                val = pmax0[key]
                val = val[np.isfinite(val)] # remove nans from failed fits
                #val = val[val>=0] # remove non-positive MLEs, these can't be log'd 
                #print(key, val)
                n, bins = np.histogram(val, normed=True)
                ax = fig.add_subplot(1,self.N_SR,pos)
                ax.plot(bins[:-1],n,drawstyle='steps-post',label="")
                ax.set_title(key)
                trueval = expected[key]
                ax.axvline(trueval,lw=2,c='k')
            plt.tight_layout()
            fig.savefig('{0}_diagnostic_null.png'.format(e.name))
            plt.close(fig)
        return dnull

analyses = {}

# We can now auto-generate the below from GAMBIT. Will be better to create a nice method of interfacing,
# like some way to save the below information to a HDF5 file and extract it here,
# but even this much is super useful, and good enough for now.

a = Analysis("ATLAS_13TeV_MultiLEP_36invfb")
a.SR_names = ["SR2_SF_loose", "SR2_SF_tight", "SR2_DF_100", "SR2_DF_150", "SR2_DF_200", "SR2_DF_300", "SR2_int", "SR2_high", "SR2_low", "SR3_slep_a", "SR3_slep_b", "SR3_slep_c", "SR3_slep_d", "SR3_slep_e", "SR3_WZ_0Ja", "SR3_WZ_0Jb", "SR3_WZ_0Jc", "SR3_WZ_1Ja", "SR3_WZ_1Jb", "SR3_WZ_1Jc", ]
a.SR_n     = [153, 9, 78, 11, 6, 2, 2, 0, 11, 4, 3, 9, 0, 0, 21, 1, 2, 1, 3, 4, ]
a.SR_b     = [133, 9.8, 68, 11.5, 2.1, 0.6, 4.1, 1.6, 4.2, 2.2, 2.8, 5.4, 1.4, 1.1, 21.7, 2.7, 1.6, 2.2, 1.8, 1.3, ]
a.SR_b_sys = [22, 2.9, 7, 3.1, 1.9, 0.6, 2.6, 1.6, 3.4, 0.8, 0.4, 0.9, 0.4, 0.2, 2.9, 0.5, 0.3, 0.5, 0.3, 0.3, ]
analyses[a.name] = a

a = Analysis("CMS_13TeV_1LEPbb_36invfb")
a.SR_names = ["SRA", "SRB", ]
a.SR_n     = [11, 7, ]
a.SR_b     = [7.5, 8.7, ]
a.SR_b_sys = [2.5, 2.2, ]
analyses[a.name] = a

a = Analysis("CMS_13TeV_2LEPsoft_36invfb")
a.SR_names = ["SR1", "SR3", "SR4", "SR5", "SR6", "SR7", "SR8", "SR9", "SR10", "SR11", "SR12", ]
a.SR_n     = [2, 19, 18, 1, 0, 3, 1, 2, 1, 2, 0, ]
a.SR_b     = [3.5, 17, 11, 1.6, 3.5, 2, 0.51, 1.4, 1.5, 1.5, 1.2, ]
a.SR_b_sys = [1, 2.4, 2, 0.7, 0.9, 0.7, 0.52, 0.7, 0.6, 0.8, 0.6, ]
analyses[a.name] = a

a = Analysis("CMS_13TeV_2OSLEP_36invfb")
a.SR_names = ["SR-2", "SR-3", "SR-4", "SR-5", "SR-7", "SR-8", "SR-9", ]
a.SR_n     = [57, 29, 2, 0, 9, 5, 1, ]
a.SR_b     = [54.9, 21.6, 6, 2.5, 7.6, 5.6, 1.3, ]
a.SR_b_sys = [7, 5.6, 1.9, 0.9, 2.8, 1.6, 0.4, ]
a.cov = [[52.8, 12.7,    3,  1.2,  4.5,  5.1,  1.2],
 [12.7, 41.4,  3.6,    2,  2.5,    2,  0.7],
 [   3,  3.6,  1.6,  0.6,  0.4,  0.3,  0.1],
 [ 1.2,    2,  0.6,  1.1,  0.3,  0.1,  0.1],
 [ 4.5,  2.5,  0.4,  0.3,  6.5,  1.8,  0.4],
 [ 5.1,    2,  0.3,  0.1,  1.8,  2.4,  0.4],
 [ 1.2,  0.7,  0.1,  0.1,  0.4,  0.4,  0.2]]
analyses[a.name] = a

a = Analysis("CMS_13TeV_2OSLEP_confnote_36invfb_NOCOVAR_NOLIKE")
a.SR_names = ["SR-1", "SR-2", "SR-3", "SR-4", "SR-5", "SR-6", "SR-7", "SR-8", "SR-9", ]
a.SR_n     = [793, 57, 29, 2, 0, 82, 9, 5, 1, ]
a.SR_b     = [793, 54.9, 21.6, 6, 2.5, 82, 7.6, 5.6, 1.3, ]
a.SR_b_sys = [32.2, 7, 5.6, 1.9, 0.9, 9.5, 2.8, 1.6, 0.4, ]
analyses[a.name] = a

a = Analysis("CMS_13TeV_MONOJET_36invfb")
a.SR_names = ["sr-0", "sr-1", "sr-2", "sr-3", "sr-4", "sr-5", "sr-6", "sr-7", "sr-8", "sr-9", "sr-10", "sr-11", "sr-12", "sr-13", "sr-14", "sr-15", "sr-16", "sr-17", "sr-18", "sr-19", "sr-20", "sr-21", ]
a.SR_n     = [136865, 74340, 42540, 25316, 15653, 10092, 8298, 4906, 2987, 2032, 1514, 926, 557, 316, 233, 172, 101, 65, 46, 26, 31, 29, ]
a.SR_b     = [134500, 73400, 42320, 25490, 15430, 10160, 8480, 4865, 2970, 1915, 1506, 844, 526, 325, 223, 169, 107, 88.1, 52.8, 25, 25.5, 26.9, ]
a.SR_b_sys = [3700, 2000, 810, 490, 310, 170, 140, 95, 49, 33, 32, 18, 14, 12, 9, 8, 6, 5.3, 3.9, 2.5, 2.6, 2.8, ]
analyses[a.name] = a

a = Analysis("CMS_13TeV_MultiLEP_36invfb")
a.SR_names = ["SR1", "SR3", "SR4", "SR5", "SR6", "SR7", "SR8", ]
a.SR_n     = [13, 19, 128, 18, 2, 82, 166, ]
a.SR_b     = [12, 19, 142, 22, 1.2, 109, 197, ]
a.SR_b_sys = [3, 4, 34, 5, 0.6, 28, 42, ]
analyses[a.name] = a

analyses["ATLAS_13TeV_MultiLEP_36invfb"].SR_s     = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 1, 0, 1, 4, ]
analyses["CMS_13TeV_1LEPbb_36invfb"].SR_s     = [0, 8, ]
analyses["CMS_13TeV_2LEPsoft_36invfb"].SR_s     = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ]
analyses["CMS_13TeV_2OSLEP_36invfb"].SR_s     = [0, 1, 3, 2, 0, 0, 0, ]
analyses["CMS_13TeV_2OSLEP_confnote_36invfb_NOCOVAR_NOLIKE"].SR_s     = [0, 0, 0, 1, 1, 0, 0, 0, 0, ]
analyses["CMS_13TeV_MONOJET_36invfb"].SR_s     = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
analyses["CMS_13TeV_MultiLEP_36invfb"].SR_s     = [0, 2, 1, 2, 1, 1, 0, ]

experiments = {}
for a in analyses.values():
   experiments[a.name] = a.make_experiment()
   experiments[a.name].N_SR = a.N_SR #useful to know this
