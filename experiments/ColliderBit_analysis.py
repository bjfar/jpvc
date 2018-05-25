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
import copy
from .experiment import Experiment

# Function to map signal hypothesis into Poisson distribution parameters
# Signal+background systematics dealt with via a MULTIPLICATIVE factor
def poisson_f_mult(s, b, theta):
    l = theta*np.atleast_1d(s + b)
    #m = (l<0)
    #l[m] = 0  #Poisson cannot have negative mean
    return {'mu': l}

# Function to map signal hypothesis into Poisson distribution parameters
# Signal+background systematics dealt with via an ADDITIVE factor
def poisson_f_add(s, b, theta):
    l = np.atleast_1d(s + b + theta)
    #m = (l<0)
    #l[m] = 0  #Poisson cannot have negative mean
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
           """Gets seeds for s and theta fits (additive nuisance)
              Gives exact MLEs for gof case!
           """
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
        def get_seeds_full(samples,full_fixed_pars):
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
        def get_seeds_null(samples,signal):
           """Gets seeds for (both additive and multiplicative) nuisance parameters fits"""
           theta_seeds={}
           theta_samples = samples[:,0,self.N_SR:].T
           for i in range(self.N_SR):
              theta_MLE = theta_samples[i]
              theta_seeds['theta_{0}'.format(i)] = theta_MLE
           return theta_seeds
        return get_seeds_null

    def seeds_null_f_gof(self): 
        def get_seeds_null(samples,signal):
            """Gets seeds for additive nuisance parameters fits
               Gives exact MLEs for gof case!
               Depends on signal hypothesis though, so calculation
               is a bit more involved."""
            verbose = False
            theta_seeds={}
            self.theta_both={} # For debugging, need to compare BOTH solutions to numerical MLEs.
            self.theta_dat={}
            bin_samples = samples[:,0,:self.N_SR].T
            theta_samples = samples[:,0,self.N_SR:].T
            for i in range(self.N_SR):
                if verbose: print("Finding MLEs for theta_{0}; total samples: {1}".format(i,len(samples)))
                s = signal['s_{0}'.format(i)] #self.SR_s[i]
                b = self.SR_b[i]
                bsys = self.SR_b_sys[i]
                n = bin_samples[i]
                x = theta_samples[i]
                A = 1./bsys**2
                B = 1 + A*(s + b - x)
                C = (s+b)*(1-A*x) - n
                D = B**2 - 4*A*C
                #D<0:
                theta_MLE = np.zeros(len(samples))
                # No solutions! This is a drag, means MLE is on boundary of allowed space, not at a 'real' minima.
                # Will mess up Wilk's theorem a bit. However MLE will necessarily just be on the boundary
                # s + b + theta = 0
                # so it is easy to compute at least
                theta_MLE[D<0] = -(s+b)
                if verbose: print("No solution count:", np.sum(D<0))
                #elif D==0:
                # One real solution:
                theta_MLE[D==0] = -B[D==0]/(2*A) # Is this always positive? Not sure...
                if verbose: print("Single solution count:", np.sum(D==0))
                #elif D>0:
                # Two real solutions
                # Hmm not sure how to pick... I guess first check that they are indeed in the "allowed" region
                if verbose: print("Two solution count:", np.sum(D>0))
                r1 = (-B + np.sqrt(D))/(2*A)
                r2 = (-B - np.sqrt(D))/(2*A)
                a1 = (s+b+r1 >= 0)
                a2 = (s+b+r2 >= 0) 
                # See if just one is allowed
                ma= (D>0) & a1 & a2
                mf= (D>0) & (~a1) & (~a2)
                # m = ma | mf 
                if verbose: print("   both allowed:", np.sum(ma))
                if verbose: print("   both forbidden:", np.sum(mf))
                # #print(list(zip(n[D>0]-s-b,r1,r2)))
                # #if both allowed, pick the one closest to the MLE for theta that one would get without the nuisance constraint
                # Actually just throw error, I think this should not happen
                #if np.sum(ma)>0:
                #    for r1i, r2i in zip(r1[ma],r2[ma]):
                #        print("r1: {0}, r2: {1}, s+b: {2}".format(r1i,r2i,s+b))
                #    raise ValueError("Found multiple allowed solutions for some MLEs! I'm pretty sure this shouldn't happen so I'm calling it an error/bug in the calculation") 
                # If both are forbidden, make sure that it isn't just by some tiny amount due to numerical error.
                fix1 = ~a1 & (s+b+r1 >= -np.abs(r1)*1e-6)
                fix2 = ~a2 & (s+b+r2 >= -np.abs(r2)*1e-6)
                #m12 = (D>0) & fix1 & fix2 # Don't need to worry about this, means they are probably the same solution anyway
                m1  = (D>0) & mf & fix1
                m2  = (D>0) & mf & fix2
                #print("Corrected {0} positive solutions".format(np.sum(m1)))
                theta_MLE[m1] = -(s+b) + np.abs(r1[m1])*1e-6 # Make sure still positive after numerics
                theta_MLE[m2] = -(s+b) + np.abs(r2[m2])*1e-6
                if np.sum(mf & ~(fix1 | fix2) > 0):
                    for r1i, r2i in zip(r1[mf],r2[mf]):
                        print("r1: {0}, r2: {1}, s+b: {2}".format(r1i,r2i,s+b))
                    raise ValueError("Found both solutions forbidden (and unfixable) for some MLEs! I'm pretty sure this shouldn't happen so I'm calling it an error/bug in the calculation") 
                #d1 = (r1 - (n - s - b))**2
                #d2 = (r2 - (n - s - b))**2
                # Or the closest to the MLE using a normal approximation for the Poisson? And fixing variance with theta=0...
                MLE_norm = (x*A + (n-s-b)/(s+b)) / (A + 1./(s+b))
                d1 = (r1 - MLE_norm)**2
                d2 = (r2 - MLE_norm)**2
                # Use this if both solutions are allowed.
                m1 = (D>0) & a1 & a2 & (d1<d2)
                m2 = (D>0) & a1 & a2 & (d1>=d2)
                if verbose: print("   both roots allowed, but positive is closer to gaussian estimate:", np.sum(m1))
                if verbose: print("   both roots allowed, but negative is closer to gaussian estimate:", np.sum(m2))
                theta_MLE[m1] = r1[m1]
                theta_MLE[m2] = r1[m2]
                # or just smallest in magnitude?
                #d1 = r1**2
                #d2 = r2**2
                # The most proper thing is just to see which solution gives the best likelihood (or minimum log-likelihood)
                #d1 = -sps.poisson.logpmf(n,s+b+r1) - sps.norm.logpdf(x,r1)
                #d2 = -sps.poisson.logpmf(n,s+b+r2) - sps.norm.logpdf(x,r2)
                #d1[~np.isfinite(d1)] = 1e99 # bad likelihood
                #d2[~np.isfinite(d2)] = 1e99
                #theta_MLE[(D>0) & (d1<d2)] = r1[(D>0) & (d1<d2)]
                #theta_MLE[(D>0) & (d1>=d2)] = r2[(D>0) & (d1>=d2)]
                #print("r1", "r2", "MLE_guess")
                #for j in range(np.sum(D>0)):
                #    print(r1[j], r2[j], MLE_norm[j])
                #theta_MLE[D>0] = MLE_norm[D>0] # test...
                # Seems like the positive root is always the right one for some reason...
                #theta_MLE[D>0] = r1[D>0] 
                # # If both solutions are forbidden then something weird is probably going on, but we'll pick one anyway
                # # and let the parameter mapping sort it out.
                m = (D>0) & (a1) & (~a2)
                if verbose: print("   only positive root allowed:", np.sum(m))
                theta_MLE[m] = r1[m]
                m = (D>0) & (~a1) & (a2)
                if verbose: print("   only negative root allowed:", np.sum(m))
                theta_MLE[m] = r2[m]
                # Save some extra info for diagnositic functions
                self.theta_both['theta_{0}'.format(i)] = [copy.copy(theta_MLE), copy.copy(theta_MLE), MLE_norm, s + b] # Make sure to copy...
                self.theta_both['theta_{0}'.format(i)][0][D>0] = r1[D>0]
                self.theta_both['theta_{0}'.format(i)][1][D>0] = r2[D>0]
                self.theta_dat['theta_{0}'.format(i)] = n,x,s,b,bsys
                # Output!
                theta_seeds['theta_{0}'.format(i)] = theta_MLE
                # Sanity check; are all these seeds part of the allowed parameter space?
                acheck = ((s + b + theta_MLE)<0)
                if np.sum(acheck)>0:
                     for soli, r1i, r2i in zip(theta_MLE[acheck],r1[acheck],r2[acheck]):
                        print("MLE: {0}, r1: {1}, r2: {2}, s+b: {3}, ? {4}".format(soli,r1i,r2i,s+b,s+b+r1i >= -np.abs(r1i)*1e-6))
                     raise ValueError("Computed {0} forbidden seeds (from {1} samples)! There is therefore a bug in the seed calculations".format(np.sum(acheck),len(samples)))
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
                         null_seeds=self.seeds_null_f(),
                         scale_with_mu=['s_{0}'.format(i) for i in range(self.N_SR)],
                         test_signal=self.test_signal
                         )
        return e


    def make_experiment_nocov(self):
        # Create the transformed pdf functions
        # Also requires some parameter renaming since we use the
        # same underlying function repeatedly
        # poisson_part_mult = [jtd.TransDist(sps.poisson,partial(poisson_f_mult,b=self.SR_b[i]),
        #                        ['s_{0} -> s'.format(i), 
        #                         'theta_{0} -> theta'.format(i)])
        #                  for i in range(self.N_SR)]

        poisson_part_add = [jtd.TransDist(sps.poisson,partial(poisson_f_add,b=self.SR_b[i]),
                               ['s_{0} -> s'.format(i), 
                                'theta_{0} -> theta'.format(i)])
                         for i in range(self.N_SR)]

        # Using lognormal constraint on multiplicative systematic parameter
        # sys_dist_mult = [jtd.TransDist(sps.lognorm,
        #                           partial(func_nuis_lognorm_mult,
        #                                   theta_std=self.SR_b_sys[i]/self.SR_b[i]),
        #                           ['theta_{0} -> theta'.format(i)])
        #               for i in range(self.N_SR)]

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
       
        nuis_options = {**theta_opt, **theta_opt2} #, 'print_level':1}
        general_options = {**s_options, **nuis_options}

        # Full observed data list, included observed values of nuisance measurements
        observed_data = np.concatenate([np.array(self.SR_n),np.zeros(self.N_SR)],axis=-1)

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
                          null_seeds=(self.seeds_null_f_gof(), True),
                          full_seeds=(self.seeds_full_f_add(), True), # Extra flag indicates that the "seeds" are actually the analytically exact MLEs, so no numerical minimisation needed
                          diagnostics=[self.make_dfull(s_opt,theta_opt),
                                       self.make_dnull(theta_opt),
                          ])
                          #             self.make_seedcheck(),
                          #             self.make_checkpdf()]
                          #)
        
        e.define_mu_test(nuisance_par_null=theta_opt,
                         null_options=nuis_options,
                         null_seeds=self.seeds_null_f(),
                         scale_with_mu=['s_{0}'.format(i) for i in range(self.N_SR)],
                         test_signal=self.test_signal
                         )

        # Just check that pdf calculation gives expected answer:
        # pars = {**s_opt,**theta_opt}
        # x = np.zeros(self.N_SR)
        # logpdf = e.general_model.logpdf(pars,e.observed_data)
        # expected_logpdf = [sps.poisson.logpmf(self.SR_n[i],self.SR_b[i]+pars['s_{0}'.format(i)]+pars['theta_{0}'.format(i)]) for i in range(self.N_SR)] \
        #                   + [sps.norm.logpdf(x[i],loc=pars['theta_{0}'.format(i)],scale=self.SR_b_sys[i]) for i in range(self.N_SR)]
        # print('logpdf         :',logpdf)
        # print('expected logpdf:', np.sum(expected_logpdf))

        # print("Components:")
        # for l, el in zip(e.general_model.logpdf_list(pars,e.observed_data), expected_logpdf):
        #     print('   logpdf:{0},  exp:{1}'.format(l[0][0],el))

        return e

    def make_dfull(self,s_opt,theta_opt):
        # Can define extra calculations to be done or plots to be created using the fit
        # results, to help diagnose any problems with the fits. 
        def dfull(e, Lmax0, pmax0, seeds0, Lmax, pmax, seeds, samples):
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
        def dnull(e, Lmax0, pmax0, seeds0, Lmax, pmax, seeds, samples):
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

    def make_seedcheck(self):
        def dseed(e, Lmax0, pmax0, seeds0, Lmax, pmax, seeds, samples):
            # Compare seeds to MLEs
            print("Comparing seeds to final MLEs for null hypothesis")
            solnames = ['seed','MLE ','r1  ','r2  ','rn  ']
            relogpdf = {n:np.zeros(len(samples)) for n in solnames}
            for key in seeds0.keys():
                print("  {0}:".format(key))
                #print(seeds0[key])
                #print(pmax0[key])
                for i,(seed, MLE) in enumerate(zip(np.atleast_1d(seeds0[key]), np.atleast_1d(pmax0[key]))):
                    r1, r2, rn, sb = self.theta_both[key]
                    if seed!=MLE:
                        #print("     seed:{0},  MLE:{1}".format(seed,MLE))
                        print("     seed:{0},  MLE:{1}, (r1={2}, r2={3}, rn={4}, s+b={5})".format(seed,MLE,r1[i],r2[i],rn[i],sb))
                    # Check likelihood values for each of these, make sure MLE is really the best fit!
                    n,x,s,b,b_sys = self.theta_dat['theta_{0}'.format(i)] 
                    for k,val in enumerate([seed,MLE,r1[i],r2[i],rn[i]]):
                        logpdf = sps.poisson.logpmf(n[i],mu=s+b+val) + sps.norm.logpdf(x[i],loc=val,scale=b_sys)
                        relogpdf[solnames[k]][i] += logpdf
                        print("       -2*logpdf: {0}, s+b+val: {1}, (running tot:{2})".format(-2*logpdf,s+b+val, -2*relogpdf[solnames[k]][i]))
 
            print("Combined likelihood using various parameters:")
            for i in range(len(samples)):
                print("Sample {0}".format(i))
                pmaxi = {par:val[i] for par,val in pmax0.items()}
                pars = {name:{**pmaxi} for name in solnames}
                for key in seeds0.keys(): 
                    seed = seeds0[key][i]
                    MLE  = pmax0[key][i]
                    r1, r2, rn, sb = self.theta_both[key]
                    pars['seed'][key] = seed
                    pars['MLE '][key] = MLE
                    pars['r1  '][key] = r1[i]
                    pars['r2  '][key] = r2[i]
                    pars['rn  '][key] = rn[i]
                minl = 1e99
                minn = ""
                for name in pars.keys(): 
                    logpdf = e.general_model.logpdf(pars[name],x=samples[i])
                    if -2*logpdf < minl: 
                        minn = name
                        minl = -2*logpdf
                    print('   -2*logpdf with {0}: {1} -- recalc: {2}'.format(name,-2*logpdf,-2*relogpdf[name][i]))
                print('   Minima was at {0} value'.format(minn))
            print()
            print("Comparing seeds to final MLEs for full hypothesis")
            for key in seeds.keys():
                print("  {0}:".format(key))
                for seed, MLE in zip(np.atleast_1d(seeds[key]), np.atleast_1d(pmax[key])):
                    if seed!=MLE:
                        print("     seed:{0},  MLE:{1}".format(seed,MLE))
        return dseed

    def make_checkpdf(self):
        def dpdf(e, Lmax0, pmax0, seeds0, Lmax, pmax, seeds, samples):
            # Checking the pdf calculation at the best fit points
            print("Manually recalculating logpdf values")
            solnames = ['seed','MLE ','r1  ','r2  ','rn  ']
            for i in range(len(samples)):
                print("Sample {0}".format(i))
                pmaxi = {par:val[i] for par,val in pmax0.items()}
                pars = {name:{**pmaxi} for name in solnames}
                for key in seeds0.keys(): 
                    seed = seeds0[key][i]
                    MLE  = pmax0[key][i]
                    r1, r2, rn, sb = self.theta_both[key]
                    pars['seed'][key] = seed
                    pars['MLE '][key] = MLE
                    pars['r1  '][key] = r1[i]
                    pars['r2  '][key] = r2[i]
                    pars['rn  '][key] = rn[i]
                for name in pars.keys():
                    print("Using {0} as parameter value".format(name)) 
                    logpdf = e.general_model.logpdf_list(pars[name],x=samples[i])
                    # recalculate pdf components manually
                    relogpdf = []
                    for j in range(self.N_SR):
                        t = 'theta_{0}'.format(j)
                        val = pars[name][t]
                        n,x,s,b,b_sys = self.theta_dat[t]
                        relogpdf += [sps.poisson.logpmf(n[i],mu=s+b+val)]
                    for j in range(self.N_SR):
                        t = 'theta_{0}'.format(j)
                        val = pars[name][t]
                        n,x,s,b,b_sys = self.theta_dat[t]
                        relogpdf += [sps.norm.logpdf(x[i],loc=val,scale=b_sys)]
                    for k, (c, ce) in enumerate(zip(logpdf, relogpdf)):
                        print('   component {0}:  pdf: {1},  re-pdf: {2}'.format(k,c,ce))
        return dpdf

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
#analyses[a.name] = a

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
#analyses["CMS_13TeV_2OSLEP_36invfb"].SR_s     = [0, 1, 3, 2, 0, 0, 0, ]
analyses["CMS_13TeV_2OSLEP_confnote_36invfb_NOCOVAR_NOLIKE"].SR_s     = [0, 0, 0, 1, 1, 0, 0, 0, 0, ]
analyses["CMS_13TeV_MONOJET_36invfb"].SR_s     = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
analyses["CMS_13TeV_MultiLEP_36invfb"].SR_s     = [0, 2, 1, 2, 1, 1, 0, ]

experiments = {}
for a in analyses.values():
   experiments[a.name] = a.make_experiment()
   experiments[a.name].N_SR = a.N_SR #useful to know this
