"""Classes related to defining what an 'experiment' needs to provide in order to be
   analysed in the jpvc framework"""

import numpy as np
import JMCtools.distributions as jtd
import JMCtools.models as jtm
import JMCtools.common as c
import scipy.stats as sps
from functools import partial

class Test:
    """Defines information required for doing fits (i.e. finding MLEs) for a
       particular kind of statisical test. Mostly just a box for fit
       options, plus some extra bits and pieces."""
    def __init__(self,nuisance_par_null,
                      test_pars,
                      null_options,
                      full_options,
                      null_seeds,
                      full_seeds,
                      DOF,
                      diagnostics=None):
        self.nuisance_par_null = nuisance_par_null
        self.test_pars         = test_pars
        self.null_options      = null_options
        self.full_options      = full_options
        self.null_seeds        = null_seeds
        self.full_seeds        = full_seeds
        self.DOF               = DOF
        self.diagnostics       = diagnostics

# Parameter transformation function for scaling signal parameters with 'mu'
def mu_parameter_mapping(mu,scale_with_mu,**kwargs):
    """Parameters that are 'replaced' by mu should be fixed to
       some nominal values using functools.partial, as should
       the list 'scale_with_mu'""" 
    out_pars = {}
    for parname, parval in kwargs.items():
        if parname in scale_with_mu:
            out_pars[parname] = mu * parval
        else:
            out_pars[parname] = parval
    return out_pars
 
class Experiment:
    def __init__(self,name,joint_pdf,observed,DOF):
        """Basic information:
           name
           joint pdf
           observed data
           degrees of freedom (of general model, i.e. used in gof test)
        """
        self.name = name
        self.joint_pdf = joint_pdf
        self.observed_data = np.array(observed)[np.newaxis,np.newaxis,:]
        self.DOF = DOF
        self.tests = {}
        self.general_model = jtm.ParameterModel(self.joint_pdf)

    def define_gof_test(self,nuisance_par_null,test_pars,null_options,full_options,null_seeds,full_seeds,diagnostics=None):
        """Set information related to 'goodness of fit' test.
           Required:
             Nuisance parameter default (null hypothesis) values.
             Options for null hypothesis fit (nuisance-only fit)
             Options for full fit (all free parameters)
             Function to produce good starting guesses for null fit (nuisance parameters)
             Function to produce good starting guesses for fit for all parameters
        """
        self.tests['gof'] = Test(nuisance_par_null,
                                 test_pars,
                                 null_options,
                                 full_options,
                                 null_seeds,
                                 full_seeds,
                                 DOF=self.DOF,
                                 diagnostics=diagnostics)

           
    def define_mu_test(self,nuisance_par_null,null_options,null_seeds,scale_with_mu,test_signal,diagnostics=None):
        """Set information related to testing via 'mu' signal strength
           parameter. This requires a bit different information that the 'gof'
           case since we need to know what parameters to scale using 'mu', but don't
           need options for the 'full' fit (since we can work them out here).
        """

        null_opt = {**null_options, 'mu': 0, 'fix_mu': True}
        full_opt = {**null_options, 'mu': 0.5, 'fix_mu': False, 'error_mu': 0.1}
        test_pars={**nuisance_par_null,'mu':0}

        self.tests['mu'] = Test(nuisance_par_null,
                                test_pars,
                                null_opt,
                                full_opt,
                                null_seeds,
                                null_seeds,
                                DOF=1,
                                diagnostics=diagnostics)
        self.tests['mu'].scale_with_mu = scale_with_mu # List of parameter to scale with mu
        self.tests['mu'].test_signal = test_signal

    def make_mu_model(self,signal):
        """Create ParameterModel object for fitting with mu_test"""
        if not 'mu' in self.tests.keys():
            raise ValueError("Options for 'mu' test have not been defined for experiment {0}!".format(self.name))

        # Currently we cannot apply the transform func directly to the JointDist object,
        # so we have to pull it apart, apply the transformation to eah submodel, and then
        # put it all back together.
        transformed_submodels = []
        for submodel, dim in zip(self.joint_pdf.submodels, self.joint_pdf.dims):
            args = c.get_dist_args(submodel)
            # Pull out the arguments that aren't getting scaled by mu, and replace them with mu.
            new_args = [a for a in args if a not in self.tests['mu'].scale_with_mu] + ['mu']
            # Pull out the arguments that ARE scaled by mu; we only need to provide these ones,
            # the other signal arguments are for some other submodel.
            sig_args = [a for a in args if a in self.tests['mu'].scale_with_mu]
            my_signal = {a: signal[a] for a in sig_args} # extract subset of signal that applies to this submodel
            transform_func = partial(mu_parameter_mapping,scale_with_mu=self.tests['mu'].scale_with_mu,**my_signal)
            trans_submodel = jtd.TransDist(submodel,transform_func,func_args=new_args)
            print('in make_mu_model:', trans_submodel.args)
            transformed_submodels += [(trans_submodel,dim)]
        print("new_submodels:", transformed_submodels)
        new_joint = jtd.JointDist(transformed_submodels)
        return jtm.ParameterModel(new_joint)

    def do_gof_test(self,test_parameters,samples=None):
        model = self.general_model
        # Test parameters fix the hypothesis that we are
        # testing. I.e. they fix some parameters during
        # the 'null' portion of the parameter fitting.
        extra_null_opt = {**test_parameters}
        for key in test_parameters.keys():
            extra_null_opt["fix_{0}".format(key)] = True # fix these parameters
        return self.do_test(model,'gof',samples,extra_null_opt,test_parameters)
 
    def do_mu_test(self,nominal_signal,samples=None):
        model = self.make_mu_model(nominal_signal)
        return self.do_test(model,'mu',samples)

    def do_test(self,model,test,samples=None,extra_null_opt=None,signal=None):
        """Perform selected statistical test"""

        print("Fitting experiment {0} in '{1}' test".format(self.name,test))

        # Get options for fitting routines
        null_opt = self.tests[test].null_options
        if extra_null_opt: null_opt.update(extra_null_opt)
        full_opt = self.tests[test].full_options
        DOF      = self.tests[test].DOF

        if samples != 'no_MC':
           # Get seeds for fitting routines, tailored to simulated (or any) data 
           null_seeds = self.tests[test].null_seeds
           full_seeds = self.tests[test].full_seeds

           # Check if seeds actually give exact MLEs for parameters
           try:
              nseeds, nexact = null_seeds
           except TypeError:
              nseeds, nexact = null_seeds, False
           try:
              fseeds, fexact = full_seeds
           except TypeError:
              fseeds, fexact = full_seeds, False

           # Manually force numerical minimization
           #fexact, nexact = False, False

           # Extract fixed parameter values from options
           null_fixed_pars = {}
           for par in null_opt:
               fixname = "fix_{0}".format(par)
               if fixname in null_opt.keys():
                   if null_opt[fixname]:
                       null_fixed_pars[par] = null_opt[par]

           full_fixed_pars = {}
           for par in null_opt:
               fixname = "fix_{0}".format(par)
               if fixname in full_opt.keys():
                   if full_opt[fixname]:
                       full_fixed_pars[par] = full_opt[par]
  
           # Do some fits!
           Nproc = 3
           seeds0 = nseeds(samples,signal) # null hypothesis fits depend on signal parameters
           seeds  = fseeds(samples)
           if nexact:
               print("Seeds are exact MLEs for null hypothesis; skipping minimisation")
               pmax0 = seeds0
               pmax0.update(null_fixed_pars)
               # need to loop over parameters, otherwise it will automatically evaluate
               # every set of parameters for every set of samples. We need them done
               # in lock-step.
               Nsamples = samples.shape[0]
               Lmax0 = np.zeros(Nsamples)
               for i,X in enumerate(samples):
                   if i % 50 == 0:
                       print("\r","Processed {0} of {1} samples...           ".format(i,Nsamples), end="")
                   pars = {}
                   for par, val in pmax0.items():
                       try:
                           pars[par] = val[i]
                       except TypeError:
                           pars[par] = val # Fixed parameters aren't arrays
                   Lmax0[i] = model.logpdf(pars,X)
           else:
               print("Fitting null hypothesis...")
               Lmax0, pmax0 = model.find_MLE_parallel(null_opt,samples,method='minuit',
                                                  Nprocesses=Nproc,seeds=seeds0)
           print()
           if fexact:
               pmax = seeds
               pmax.update(full_fixed_pars)
               Nsamples = samples.shape[0]
               Lmax = np.zeros(Nsamples)
               for i,X in enumerate(samples):
                   if i % 50 == 0:
                       print("\r","Processed {0} of {1} samples...           ".format(i,Nsamples), end="")
                   pars = {}
                   for par, val in pmax.items():
                       try:
                           pars[par] = val[i]
                       except TypeError:
                           pars[par] = val # Fixed parameters aren't arrays
                   Lmax[i] = model.logpdf(pars,X)
           else:
               print("Fitting alternate hypothesis (free signal)...")
               Lmax, pmax = model.find_MLE_parallel(full_opt,samples,method='minuit',
                                                Nprocesses=Nproc,seeds=seeds)
           print()

           #print(null_opt)
           #print(full_opt)
           #print("Lmax0",Lmax0)
           #print("Lmax",Lmax)

           # Run diagnostics functions for this experiment + test
           print("Running extra diagnostic functions")
           dfuncs = self.tests[test].diagnostics
           if dfuncs:
               for f in dfuncs:
                   f(self, Lmax0, pmax0, seeds0, Lmax, pmax, seeds, samples)
 
           # Likelihood ratio test statistics
           LLR = -2*(Lmax0 - Lmax)

           # Correct (hopefully) small numerical errors
           LLR[LLR<0] = 0
        else:
           LLR = None

        # Also fit the observed data so we can compute its p-value 
        print("Fitting with observed data...")
        odata = self.observed_data
        seeds0_obs = nseeds(odata,signal) # null hypothesis fits depend on signal parameters
        seeds_obs  = fseeds(odata)
        if nexact:
            pmax0_obs = seeds0_obs
            pmax0_obs.update(null_fixed_pars)
            Lmax0_obs = model.logpdf(pmax0_obs,odata[0])
        else: 
            Lmax0_obs, pmax0_obs = model.find_MLE_parallel(null_opt, odata, 
                      method='minuit', Nprocesses=1, seeds=seeds0_obs)
        if fexact:
            pmax_obs = seeds_obs
            pmax_obs.update(full_fixed_pars)
            Lmax_obs = model.logpdf(pmax_obs,odata[0])
        else: 
            Lmax_obs, pmax_obs = model.find_MLE_parallel(full_opt, odata, 
                      method='minuit', Nprocesses=1, seeds=seeds_obs)

        # Asymptotic p-value
        LLR_obs = -2 * (Lmax0_obs[0] - Lmax_obs[0])
        apval = 1 - sps.chi2.cdf(LLR_obs, DOF) 

        # Empirical p-value
        a = np.argsort(LLR)
        print("LLR:",LLR[a])
        print("Lmax0:",Lmax0[a])
        print("Lmax :",Lmax[a])
        if LLR is not None:
           epval = c.e_pval(LLR,LLR_obs)
        else:
           epval = None
        print("LLR_obs:", LLR_obs)
        #print("odata:", odata)
        return LLR, LLR_obs, apval, epval, DOF




