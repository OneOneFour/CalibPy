from gpytorch.models import ExactGP
from gpytorch.kernels import MultitaskKernel, RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean,MultitaskMean
from gpytorch.constraints import GreaterThan
from gpytorch.priors import LogNormalPrior
from gpytorch.distributions import MultitaskMultivariateNormal

class MultiTaskGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, rank=1):
        super(MultiTaskGP, self).__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks
        self.mean_module = MultitaskMean(ConstantMean(),num_tasks=num_tasks)
        self.base_covar_module = RBFKernel(lengthscale_constraint=GreaterThan(0.1),lengthscale_prior=LogNormalPrior(0, 1))
        self.covar_module = MultitaskKernel(
            self.base_covar_module, num_tasks=num_tasks, rank=rank
        )


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean=mean_x, covariance_matrix=covar_x)
