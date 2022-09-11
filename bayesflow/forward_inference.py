# Copyright (c) 2022 The BayesFlow Developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import logging
logging.basicConfig()

from bayesflow.default_settings import DEFAULT_KEYS
from bayesflow.exceptions import ConfigurationError
from bayesflow.diagnostics import plot_prior2d


class ContextGenerator:
    """ Basic interface for a simulation module responsible for generating variables over which
    we want to amortize during simulation-based training, but do not want to perform inference on.
    Both priors and simulators in a generative framework can have their own context generators,
    depending on the particular modeling goals.

    The interface distinguishes between two types of context: batchable and non-batchable.

    - Batchable context variables differ for each simulation in each training batch
    - Non-batchable context varibales stay the same for each simulation in a batch, but differ across batches

    Examples for batchable context variables include experimental design variables, design matrices, etc.
    Examples for non-batchable context variables include the number of observations in an experiment, positional
    encodings, time indices, etc.

    While the latter can also be considered batchable in principle, batching them would require non-Tensor
    (i.e., non-rectangular) data structures, which usually means inefficient computations.  

    Example for a simulation context which will generate a random number of observations between 1 and 100 for 
    each training batch:

    >>> gen = ContextGenerator(non_batchable_context_fun=lambda : np.random.randint(1, 101))
    """

    def __init__(self, batchable_context_fun : callable = None, non_batchable_context_fun: callable = None,
                 use_non_batchable_for_batchable: bool = False):
        """ Instantiates a context generator responsible for random generation of variables which vary from data set
        to data set but cannot be considered data or parameters, e.g., time indices, number of observations, etc.
        A batchable, non-batchable, or both context functions should be provided to the constructor. An optional
        argument dictates whether the outputs of the non-batchable context function should be used as inputs to
        batchable context.
        
        Parameters
        ----------
        batchable_context_fun             : callable
            A function with optional control arguments responsible for generating per-simulation set context variables
        non_batchable_context_fun         : callable 
            A function with optional control arguments responsible for generating per-batch-of-simulations context variables.
        use_non_batchable_for_batchable   : bool, optional, default: False
            Determines whether to use output of non_batchable_context_fun as input to batchable_context_fun. Only relevant
            when both context types are provided.
        """
        
        self.batchable_context_fun = batchable_context_fun
        self.non_batchable_context_fun = non_batchable_context_fun
        self.use_non_batchable_for_batchable = use_non_batchable_for_batchable

    def __call__(self, batch_size, *args, **kwargs):
        """ Wraps the method generate_context, which returns a dictionary with 
        batchable and non batchable context.
        
        Optional positional and keyword arguments are passed to the internal 
        context-generating functions or ignored if the latter are None.

         Parameters
        ----------

        batch_size : int
            The batch_size argument used for batchable context.

        Returns
        -------

        context_dict : dictionary
            A dictionary with context variables with the following keys:
            `batchable_context` : value
            `non_batchable_context` : value
        
        Note, that the values of the context variables will be None, if the
        corresponding context-generating functions have not been provided when
        initializing this object.
        """

        return self.generate_context(batch_size, *args, **kwargs)

    def batchable_context(self, batch_size, *args, **kwargs):
        """ Generates 'batch_size' context variables given optional arguments. 
        Return type is a list of context variables.
        """
        if self.batchable_context_fun is not None:
            context = [self.batchable_context_fun(*args, **kwargs) for _ in range(batch_size)]
            return context
        return None

    def non_batchable_context(self, *args, **kwargs):
        """ Generates a context variable shared across simulations in a given batch, given optional arguments.
        """
        if self.non_batchable_context_fun is not None:
            return self.non_batchable_context_fun(*args, **kwargs)
        return None

    def generate_context(self, batch_size, *args, **kwargs):
        """ Creates a dictionary with batchable and non batchable context.

         Parameters
        ----------

        batch_size : int
            The batch_size argument used for batchable context.

        Returns
        -------

        context_dict : dictionary
            A dictionary with context variables with the following keys, if default keys not changed:
            `batchable_context` : value
            `non_batchable_context` : value
        
        Note, that the values of the context variables will be None, if the
        corresponding context-generating functions have not been provided when
        initializing this object.
        """

        out_dict = {}
        out_dict[DEFAULT_KEYS['non_batchable_context']] = self.non_batchable_context()
        if self.use_non_batchable_for_batchable:
            out_dict[DEFAULT_KEYS['batchable_context']] = self.batchable_context(batch_size, 
            out_dict[DEFAULT_KEYS['non_batchable_context']], *args, **kwargs)
        else:
            out_dict[DEFAULT_KEYS['batchable_context']] = self.batchable_context(batch_size, *args, **kwargs)
        return out_dict
        

class Prior:
    """ Basic interface for a simulation module responsible for generating random draws from a 
    prior distribution.

    The prior functions should return a np.array of simulation parameters which will be internally used
    by the GenerativeModel interface for simulations.
   
    An optional context generator (i.e., an instance of ContextGenerator) or a user-defined callable object 
    implementing the following two methods can be provided:
    - context_generator.batchable_context(batch_size)
    - context_generator.non_batchable_context()
    """

    def __init__(self, batch_prior_fun : callable = None, prior_fun : callable = None, 
                 context_generator : callable = None, param_names : list = None):
        """
        Instantiates a prior generator which will draw random parameter configurations from a user-informed prior
        distribution. No improper priors are allowed, as these may render the generative scope of a model undefined.
        
        Parameters
        ----------
        batch_ prior_fun    : callable
            A function (callbale object) with optional control arguments responsible for generating batches 
            of per-simulation parameters.
        prior_fun           : callable
            A function (callbale object) with optional control arguments responsible for generating 
            per-simulation parameters.
        context generator   : callable, optional, (default None, recommended instance of ContextGenerator)
            An optional function (ideally an instance of ContextGenerator) for generating prior context variables.
        param_names         : list of str, optional, (default None)
            A list with strings representing the names of the parameters.
        """

        if (batch_prior_fun is None) is (prior_fun is None):
            raise ConfigurationError('Either batch_prior_fun or prior_fun should be provided, but not both!')
        self.prior = prior_fun
        self.batched_prior = batch_prior_fun
        self.context_gen = context_generator
        self.param_names = param_names
        if prior_fun is None:
            self.is_batched = True
        else:
            self.is_batched = False

    def __call__(self, batch_size, *args, **kwargs):
        """ Generates `batch_size` draws from the prior given optional context generator.

        Parameters
        ----------
        batch_size : int
            The number of draws to obtain from the prior + context generator functions.
        *args      : tuple
            Optional positional arguments passed to the generator functions.
        **kwargs   : dict
            Optional keyword arguments passed to the generator functions.

        Returns
        -------
        out_dict - a dictionary with the quantities generated from the prior + context funcitons.
        """

        if self.batched_prior is not None:
            self.is_batched = True

        # Prepare placeholder output dictionary
        out_dict = {
            DEFAULT_KEYS['prior_draws']: None,
            DEFAULT_KEYS['batchable_context'] : None,
            DEFAULT_KEYS['non_batchable_context'] : None
        }

        # Populate dictionary with context or leave at None
        if self.context_gen is not None:
            context_dict = self.context_gen(batch_size, *args, **kwargs)
            out_dict[DEFAULT_KEYS['non_batchable_context']] = context_dict['non_batchable_context']
            out_dict[DEFAULT_KEYS['batchable_context']] = context_dict[DEFAULT_KEYS['batchable_context']]

        # Generate prior draws according to context:
        # No context type
        if out_dict[DEFAULT_KEYS['batchable_context']] is None and out_dict[DEFAULT_KEYS['non_batchable_context']] is None:
            if self.is_batched:
                out_dict[DEFAULT_KEYS['prior_draws']] = np.array(self.batched_prior(batch_size=batch_size, *args, **kwargs))
            else:
                out_dict[DEFAULT_KEYS['prior_draws']] = np.array([self.prior(*args, **kwargs) for _ in range(batch_size)])
        
        # Only batchable context
        elif out_dict[DEFAULT_KEYS['non_batchable_context']] is None:
            if self.is_batched:
                out_dict[DEFAULT_KEYS['prior_draws']] = np.array(self.batched_prior(out_dict[DEFAULT_KEYS['batchable_context']], batch_size=batch_size, *args, **kwargs))
            else: 
                out_dict[DEFAULT_KEYS['prior_draws']] = np.array([self.prior(out_dict[DEFAULT_KEYS['batchable_context']][b], *args, **kwargs) 
                for b in range(batch_size)])
            
        # Only non-batchable context
        elif out_dict[DEFAULT_KEYS['batchable_context']] is None:
            if self.is_batched:
                out_dict[DEFAULT_KEYS['prior_draws']] = np.array(self.batched_prior(out_dict[DEFAULT_KEYS['non_batchable_context']], batch_size=batch_size))
            else:
                out_dict[DEFAULT_KEYS['prior_draws']] = np.array([self.prior(out_dict[DEFAULT_KEYS['non_batchable_context']], *args, **kwargs) 
                for _ in range(batch_size)])

        # Both batchable and non_batchable context
        else:
            if self.is_batched:
                out_dict[DEFAULT_KEYS['prior_draws']] = np.array(
                    self.batched_prior(out_dict[DEFAULT_KEYS['batchable_context']], 
                               out_dict[DEFAULT_KEYS['non_batchable_context']], batch_size=batch_size, *args, **kwargs))
            else:    
                out_dict[DEFAULT_KEYS['prior_draws']] = np.array([
                    self.prior(out_dict[DEFAULT_KEYS['batchable_context']][b], 
                               out_dict[DEFAULT_KEYS['non_batchable_context']], *args, **kwargs) 
                    for b in range(batch_size)])

        return out_dict

    def plot_prior2d(self, **kwargs):
        """ Generates a 2D plot representing bivariate prior ditributions. Uses the function
        `bayesflow.diagnostics.plot_prior2d() internally for generating the plot.

        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments passed to the `plot_prior2d` function.

        Returns
        -------
        f : plt.Figure - the figure instance for optional saving
        """

        return plot_prior2d(self, param_names=self.param_names, **kwargs)

    def estimate_means_and_stds(self, n_draws=1000, *args, **kwargs):
        """ Estimates prior means and stds given n_draws from the prior, useful
        for z-standardization of the prior draws.
        
        Parameters
        ----------

        n_draws: int, optional (default = 1000)
            The number of random draws to obtain from the joint prior.
        *args      : tuple
            Optional positional arguments passed to the generator functions.
        **kwargs   : dict
            Optional keyword arguments passed to the generator functions.

        Returns
        -------    
        (prior_means, prior_stds) - tuple of np.ndarrays
            The estimated means and stds of the joint prior.
        """

        out_dict = self(n_draws, *args, **kwargs)
        prior_means = np.mean(out_dict[DEFAULT_KEYS['prior_draws']], axis=0, keepdims=True)
        prior_stds = np.std(out_dict[DEFAULT_KEYS['prior_draws']], axis=0, ddof=1, keepdims=True)
        return prior_means, prior_stds

    def logpdf(self, prior_draws):
        raise NotImplementedError('Prior density computation is under construction!')
            

class Simulator:
    """ Basic interface for a simulation module responsible for generating randomized simulations given a prior
    parameter distribution and optional context variables, given a user-provided simulation function.

    The user-provided simulator functions should return a np.array of synthetic data which will be used internally
    by the GenerativeModel interface for simulations.
   
    An optional context generator (i.e., an instance of ContextGenerator) or a user-defined callable object 
    implementing the following two methods can be provided:
    - context_generator.batchable_context(batch_size)
    - context_generator.non_batchable_context()
    """

    def __init__(self, batch_simulator_fun=None, simulator_fun=None, context_generator=None):
        """ Instantiates a data generator which will perform randomized simulations given a set of parameters and optional context.
        Either a batch_simulator_fun or simulator_fun, but not both, should be provided to instantiate a Simulator object.

        If a batch_simulator_fun is provided, the interface will assume that the function operates on batches of parameter
        vectors and context variables and will pass the latter directly to the function. Power users should attempt to provide
        optimized batched simulators. 

        If a simulator_fun is provided, the interface will assume thatthe function operates on single parameter vectors and
        context variables and will wrap the simulator internally to allow batched functionality.
        
        Parameters
        ----------
        batch_simulator_fun  : callable
            A function (callbale object) with optional control arguments responsible for generating a batch of simulations
            given a batch of parameters and optional context variables.
        simulator_fun       : callable
            A function (callable object) with optional control arguments responsible for generating a simulaiton given
            a single parameter vector and optional variables.
        context generator   : callable (default None, recommended instance of ContextGenerator)
            An optional function (ideally an instance of ContextGenerator) for generating prior context variables.
        """

        if (batch_simulator_fun is None) is (simulator_fun is None):
            raise ConfigurationError('Either batch_simulator_fun or simulator_fun should be provided, but not both!')
        
        self.is_batched = True if batch_simulator_fun is not None else False
        
        if self.is_batched:
            self.simulator = batch_simulator_fun
        else:
            self.simulator = simulator_fun
        self.context_gen = context_generator
        
    def __call__(self, params, *args, **kwargs):
        """ Generates simulated data given param draws and optional context variables generated internally.
        
        Parameters
        ----------
        params   :  np.ndarray of shape (n_sim, ...) - the parameter draws obtained from the prior.

        Returns
        -------

        out_dict : dictionary
            An output dictionary with randomly simulated variables, the following keys are mandatory, if default keys not modified:
            `sim_data` : value
            `non_batchable_context` : value
            `batchable_context` : value
        """
        
        # Always assume first dimension is batch dimension
        batch_size = params.shape[0]
        
        # Prepare placeholder dictionary
        out_dict = {
            DEFAULT_KEYS['sim_data']: None,
            DEFAULT_KEYS['batchable_context'] : None,
            DEFAULT_KEYS['non_batchable_context'] : None
        }
        
        # Populate dictionary with context or leave at None
        if self.context_gen is not None:
            context_dict = self.context_gen.generate_context(batch_size, *args, **kwargs)
            out_dict[DEFAULT_KEYS['non_batchable_context']] = context_dict[DEFAULT_KEYS['non_batchable_context']]
            out_dict[DEFAULT_KEYS['batchable_context']] = context_dict[DEFAULT_KEYS['batchable_context']]
        
        if self.is_batched:
            return self._simulate_batched(params, out_dict, *args, **kwargs)
        return self._simulate_non_batched(params, out_dict, *args, **kwargs)
        
    def _simulate_batched(self, params, out_dict, *args, **kwargs):
        """ Assumes a batched simulator accepting batched contexts and priors.
        """
        
        # No context type
        if out_dict[DEFAULT_KEYS['batchable_context']] is None and out_dict[DEFAULT_KEYS['non_batchable_context']] is None:
            out_dict[DEFAULT_KEYS['sim_data']] = self.simulator(params, *args, **kwargs)
            
        # Only batchable context
        elif out_dict['non_batchable_context'] is None:
            out_dict[DEFAULT_KEYS['sim_data']] = self.simulator(params, 
                                                  out_dict[DEFAULT_KEYS['batchable_context']], *args, **kwargs)

        # Only non-batchable context
        elif out_dict[DEFAULT_KEYS['batchable_context']] is None:
            out_dict[DEFAULT_KEYS['sim_data']] = self.simulator(params, 
                                                  out_dict[DEFAULT_KEYS['non_batchable_context']], *args, **kwargs)
        
        # Both batchable and non-batchable context
        else:
            out_dict[DEFAULT_KEYS['sim_data']] = self.simulator(params, 
                                                  out_dict[DEFAULT_KEYS['batchable_context']], 
                                                  out_dict[DEFAULT_KEYS['non_batchable_context']], *args, **kwargs)

        return out_dict
    
    def _simulate_non_batched(self, params, out_dict, *args, **kwargs):
        """ Assumes a non-batched simulator accepting batched contexts and priors.
        """
        
        # Extract batch size
        batch_size = params.shape[0]
        
        # No context type
        if out_dict[DEFAULT_KEYS['batchable_context']] is None and out_dict[DEFAULT_KEYS['non_batchable_context']] is None:
            out_dict[DEFAULT_KEYS['sim_data']] = np.array([self.simulator(params[b],  *args, **kwargs) for b in range(batch_size)])
            
        # Only batchable context
        elif out_dict['non_batchable_context'] is None:
            out_dict[DEFAULT_KEYS['sim_data']] = np.array([self.simulator(params[b], 
                                                            out_dict[DEFAULT_KEYS['batchable_context']][b], 
                                                            *args, **kwargs) 
                                             for b in range(batch_size)])
            
        # Only non-batchable context
        elif out_dict[DEFAULT_KEYS['batchable_context']] is None:
            out_dict[DEFAULT_KEYS['sim_data']] = np.array([self.simulator(params[b], 
                                                            out_dict[DEFAULT_KEYS['non_batchable_context']], 
                                                            *args, **kwargs) 
                                             for b in range(batch_size)])
            
        # Both batchable and non_batchable context
        else:
            out_dict[DEFAULT_KEYS['sim_data']] = np.array([self.simulator(params[b], 
                                                            out_dict[DEFAULT_KEYS['batchable_context']][b], 
                                                            out_dict[DEFAULT_KEYS['non_batchable_context']], 
                                                            *args, **kwargs) 
                                             for b in range(batch_size)])

        return out_dict
                

class GenerativeModel:
    """
    Basic interface for a generative model in a simulation-based context.
    Generally, a generative model consists of two mandatory components:
    
    - Prior : A randomized function returning random parameter draws from a prior distribution;
    - Simulator : A function which transforms the parameters into observables in a non-deterministic manner.
    """

    _N_SIM_TEST = 2 
    
    def __init__(self, prior: callable, simulator: callable, skip_test: bool = False, 
                 prior_is_batched: bool = False, simulator_is_batched: bool = False, name: str = "anonymous"):
        """
        Instantiates a generative model responsible for drawing generating params, data, and optional context.
        
        Parameters
        ----------
        prior                : callable or bayesflow.forward_inference.Prior instance
            A function returning random draws from the prior parameter distribution. Should encode
            prior knowledge about plausible parameter ranges;
        simulator            : callable or bayesflow.forward_inference.Simulator instance
            A function accepting parameter draws, optional context, and optional arguments as input
            and returning obseravble data;
        skip_test            : bool (default - False)
            If True, a forward inference pass will be performed.
        prior_is_batched     : bool (default - False), only relevant and mandatory if providing a custom prior without
            the Prior wrapper.  
        simulator_is_batched : bool (default - False), only relevant and mandatory if providing a custom simulator without
            the Simulator wrapper. 
        name                 : str (default - "anonoymous")
            An optional name for the generative model. If kept default (None), 'anonymous' is set as name.

        Important
        ----------
        If you are not using the provided Prior and Simulator wrappers for your prior and data generator,
        only functions returning a np.ndarray in the correct format will be accepted, since these will be
        wrapped internally. In addition, you need to indicate whether your simulator operates on batched of
        parameters or on single parameter vectors via tha `simulator_is_batched` argument.
        """
        
        if type(prior) is not Prior:
            prior_args = {'prior_fun': prior} if prior_is_batched else {'prior_batch_fun': prior}
            self.prior = Prior(prior_fun=prior)
            self.prior_is_batched = prior_is_batched
        else:
            self.prior = prior
            self.prior_is_batched = prior_is_batched

        if type(simulator) is not Simulator:
            self.simulator = self._config_custom_simulator(simulator, simulator_is_batched)
        else:
            self.simulator = simulator
            self.simulator_is_batched = self.simulator.is_batched
        

        if name is None:
            self.name = 'anonymous'
        else:
            self.name = name
        self.param_names = prior.param_names
        
        if not skip_test:
            self._test()

    def __call__(self, batch_size, *args, **kwargs):
        """ Carries out forward inference 'batch_size' times.
        """

        # Forward inference
        prior_out = self.prior(batch_size, *args, **kwargs)
        sim_out = self.simulator(prior_out['prior_draws'], *args, **kwargs)

        # Prepare and fill placeholder dict
        out_dict = {
            DEFAULT_KEYS['prior_non_batchable_context']: prior_out[DEFAULT_KEYS['non_batchable_context']],
            DEFAULT_KEYS['prior_batchable_context']: prior_out[DEFAULT_KEYS['batchable_context']],
            DEFAULT_KEYS['prior_draws']:  prior_out[DEFAULT_KEYS['prior_draws']],
            DEFAULT_KEYS['sim_non_batchable_context']: sim_out[DEFAULT_KEYS['non_batchable_context']],
            DEFAULT_KEYS['sim_batchable_context']: sim_out[DEFAULT_KEYS['batchable_context']],
            DEFAULT_KEYS['sim_data']: sim_out[DEFAULT_KEYS['sim_data']]
        }

        return out_dict

    def _config_custom_simulator(self, sim_fun, is_batched):
        """ Only called if user has provided a custom simulator not using the Simulator wrapper.
        """

        if is_batched is None:
            raise ConfigurationError('Since you are not using the Simulator wrapper, please set ' +
                                     'simulator_is_batched to True if your simulator operates on batches, ' +
                                     'otherwise set it to False.')
        elif is_batched:
            return Simulator(batch_simulator_fun=sim_fun)
        else:
            return Simulator(simulator_fun=sim_fun)

    def plot_prior_pushforward(self, funcs_list):
        """ TODO"""
        raise NotImplementedError('Prior density computation is under construction!')

    def _test(self):
        """ Performs a sanity check on forward inference and some verbose information.
        """

        # Use minimal n_sim > 1
        _n_sim = GenerativeModel._N_SIM_TEST
        out = self(_n_sim)

        # Logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Attempt to log batch results or fail and warn user
        try:
            logger.info(f'Performing {_n_sim} pilot runs with the {self.name} model...')
            # Format strings
            p_shape_str = "(batch_size = {}, -{}".format(out[DEFAULT_KEYS['prior_draws']].shape[0], 
                                                         out[DEFAULT_KEYS['prior_draws']].shape[1:])
            p_shape_str = p_shape_str.replace('-(', '').replace(',)', ')')
            d_shape_str = "(batch_size = {}, -{}".format(out[DEFAULT_KEYS['sim_data']].shape[0], 
                                                         out[DEFAULT_KEYS['sim_data']].shape[1:])
            d_shape_str = d_shape_str.replace('-(', '').replace(',)', ')')

            # Log to default-config
            logger.info(f'Shape of parameter batch after {_n_sim} pilot simulations: {p_shape_str}')
            logger.info(f'Shape of simulation batch after {_n_sim} pilot simulations: {d_shape_str}')

            for k, v in out.items():
                if 'context' in k:
                    name = k.replace('_', ' ').replace('sim', 'simulation').replace('non ', 'non-')
                    if v is None:
                        logger.info(f'No optional {name} provided.')
                    else:
                        try:
                            logger.info(f'Shape of {name}: {v.shape}')
                        except Exception as _:
                            logger.info(f'Could not determine shape of {name}. Type appears to be non-array: {type(v)},\
                                    so make sure your input configurator takes cares of that!')
        except Exception as err:
            raise ConfigurationError('Could not run forward inference with specified generative model...' +
                                    f'Please re-examine model components!\n {err}')
            

class MultiGenerativeModel:
    """ Basic interface for multiple generative models in a simulation-based context.
    A MultiveGenerativeModel instance consists of a list of GenerativeModel instances
    and a prior distribution over candidate models defined by a list of probabilities.
    """

    def __init__(self, generative_models: list, model_probs='equal'):
        """
        Instantiates a multi-generative model responsible for generating parameters, data, and optional context
        from a list of models according to specified prior model probabilities (PMPs).
        
        Parameters
        ----------
        generative_models : list of GenerativeModel instances
            The list of candidate generative models
        model_probs       : string (default - 'equal') or list of floats with sum(model_probs) == 1.
            The list of model probabilities, should have the same length as the list of
            generative models. Note, that probabilities should sum to one.
        """

        self.generative_models = generative_models
        self.n_models = len(generative_models)
        self.model_prior = self._determine_model_prior(model_probs)

    def _determine_model_prior(self, model_probs):
        """Creates the model prior p(M) given user input."""

        if model_probs == 'equal':
            return lambda b: np.random.randint(self.n_models, size=b)
        return lambda b: np.random.default_rng().choice(self.n_models, size=b, p=model_probs)
        
    def __call__(self, batch_size, **kwargs):
        
        # Prepare placeholders
        out_dict = {
            DEFAULT_KEYS['model_outputs']: [],
            DEFAULT_KEYS['model_indices']: []
        }
        
        # Sample model indices
        model_indices = self.model_prior(batch_size)

        # gather model indices and simulate datasets of same model index as batch
        # create frequency table of model indices
        m_idx, n = np.unique(model_indices, return_counts=True)

        # Iterate over each unique model index and create all data sets for that model index
        for m_idx, n in zip(m_idx, n):

            model_out = self.generative_models[m_idx](n, **kwargs)
            out_dict[DEFAULT_KEYS['model_outputs']].append(model_out)
            out_dict[DEFAULT_KEYS['model_indices']].append(m_idx)
        
        return out_dict