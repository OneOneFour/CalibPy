from typing import List

import torch
import torch.distributions


class LatinHypercubeSampler:
    """
    A class used to perform Latin Hypercube Sampling (LHS) from a list of distributions.

    Attributes:
        distributions (List[distributions.Distribution]): A list of PyTorch distribution objects from which to sample. Distributions are assumed to be independent.
        d (int): The number of distributions.
        centered (bool): Whether to center the samples within each interval.

    Methods:
        sample(n: int) -> torch.Tensor:
            Generates n samples from the specified distributions using LHS.
    """

    def __init__(
        self,
        distributions: List[torch.distributions.Distribution],
        centered: bool = False,
    ):
        """
        Constructs all the necessary attributes for the LatinHypercubeSampler object.

        Parameters:
            distributions (List[torch.distributions.Distribution]): A list of PyTorch distribution objects from which to sample.
            centered (bool): Whether to center the samples within each interval. Defaults to False.
        """
        self.distributions = distributions
        self.d = len(distributions)
        self.centered = centered

    def sample(self, n: int) -> torch.Tensor:
        """
        Generates n samples from the specified distributions using Latin Hypercube Sampling.

        Parameters:
            n (int): The number of samples to generate.

        Returns:
            torch.Tensor: A tensor of shape (n, d) containing the generated samples.
        """
        samples = torch.zeros((n, self.d))
        # Work in uniform space
        for i in range(self.d):
            uniform_cdf = (
                torch.randperm(n)
                + (1 - int(self.centered)) * torch.rand(n)
                + int(self.centered) * 0.5
            ) / n
            # Use icdf method on distribution to transform to sample values
            samples[:, i] = self.distributions[i].icdf(uniform_cdf)
        return samples
