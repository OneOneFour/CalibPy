from typing import Callable, Optional

import torch

from .calibration import ApproxBayesianMethod


class ApproximateBayesianComputation(ApproxBayesianMethod):
    """
    Directly draw samples from posterior p(theta|y).
    Such a method can be good for a relatively cheap, low dimensional f(\theta) + \epsilon = y problem.
    ABC scales poorly with dimensionality as well as the skill of the summary statistic leading to an exponential
    slow down.
    """

    def __init__(
        self, parameters, priors, observables, summary, tolerance, forward=None
    ):
        super().__init__(parameters, priors, observables, summary, forward)
        self.tolerance = tolerance
        self.accepted_samples = None

    def sample_posterior(
        self,
        n_samples: int,
        verbose: bool = False,
        batch_params: Optional[torch.Size] = None,
    ):
        if self.forward is None:
            raise RuntimeError("Forward model must be provided to calibrate directly")
        nsamples = 0
        for thetas in self.iterate(n_samples, batch_params=batch_params):
            observations = self.forward(thetas)
            nsamples += batch_params[0] if batch_params else 1
            self.step(thetas, observations)
            if verbose:
                message = (
                    f"\nCurrent Statistics\n"
                    f"Number of Accepted Samples: {len(self.accepted_samples)}\n"
                    f"Number of Trials: {nsamples}\n"
                    f"Acceptance Rate: {(len(self.accepted_samples))/nsamples:.2%}"
                )
                print(message, end="\r", flush=True) 
                print("\033[F" * message.count("\n"), end="")



        if verbose:
            print(
                f"\n\n\n\n\n\nFinal Statistics\nNumber of Accepted Samples:{len(self.accepted_samples)}\n"
                f"Number of Trials:{nsamples}\n"
                f"Acceptance Rate:{(len(self.accepted_samples))/nsamples:.2%}"
            )
        return self.accepted_samples
    
    def kde_posterior(self, theta):
        """
        Compute the posterior using Kernel Density Estimation
        """
        raise NotImplementedError


    def step(self, samples: torch.Tensor, observations: torch.Tensor):
        """
        Perform a calibration step. Takes in new observations
        """
        summaries = self.summary(observations)
        accept = summaries <= self.tolerance
        self.accepted_samples = torch.cat([self.accepted_samples, samples[accept]])
        return samples[accept]

    def iterate(self, n_samples: int, batch_params: Optional[torch.Size] = None,args_as_dict:bool = False):
        """
        Perform n_iter iterations of ABC
        """
        self.accepted_samples = torch.empty([0, self.number_of_parameters])
        while len(self.accepted_samples) < n_samples:
            if batch_params is not None:
                thetas = self.sample_prior(batch_params, labeled=args_as_dict)
                yield thetas
            else:
                theta = self.sample_prior(labeled=args_as_dict)
                yield theta
        if batch_params:
            self.accepted_samples = self.accepted_samples[:n_samples]
