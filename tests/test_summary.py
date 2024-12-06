import torch
from torch.distributions import Normal, MultivariateNormal
from pytest import approx
from gpytorch.distributions import MultitaskMultivariateNormal
from abracalibra.summary import (
    l2_norm,
    l1_norm,
    linf_norm,
    implausibility_univariate_distr,
    implausibility_multivariate_distr,
    batched_implausibility_univariate_distr
)


def test_l2norm():
    x = torch.tensor([[10.0, 5.0]])
    y = torch.tensor([[7.0, 1.0]])
    result = l2_norm(x, y)
    assert torch.isclose(result, torch.tensor(5.0))


def test_l1norm():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    result = l1_norm(x, y)
    assert torch.isclose(result, torch.tensor(4.0))


def test_linf_norm():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([5.0, 4.0])
    result = linf_norm(x, y)
    assert result == approx(torch.tensor(4.0))


def test_implausibility_univariate_distr():
    data = Normal(0.0, 1.0)
    prediction = MultivariateNormal(torch.tensor([1.0]), torch.eye(1))
    result = implausibility_univariate_distr(prediction, data).squeeze()
    true_implausibility = torch.sqrt(torch.tensor(2)) / 2  # true result is 1/√2
    assert result == approx(true_implausibility)


def test_max_implausibility_over_univariate_data():
    datas = [Normal(0.0, 1.0), Normal(0.0, 1.0), Normal(0.0, 1.0)]
    predictions = [
        Normal(torch.tensor([1.0]), torch.eye(1)),
        Normal(torch.tensor([0.5]), torch.eye(1)),
        Normal(torch.tensor([0.2]), torch.eye(1)),
    ]
    results = torch.tensor([ implausibility_univariate_distr(prediction, data) for prediction, data in zip(predictions, datas)])  
    result = torch.max(results)
    true_implausibility = torch.sqrt(torch.tensor(2)) / 2  # true result is 1/√2   
    assert result == approx(true_implausibility)


def test_implausibility_univariate_batched_1d():
    datas = [Normal(0.0, 1.0), Normal(0.0, 1.0), Normal(0.0, 1.0)]
    batched_data = Normal(torch.stack([d.mean for d in datas]), torch.stack([d.variance for d in datas])**0.5)   
    predictions = MultivariateNormal(torch.stack([torch.ones(5),torch.zeros(5),torch.zeros(5)]),torch.stack([torch.eye(5),torch.eye(5),torch.eye(5)]))
    
    results = torch.max(batched_implausibility_univariate_distr(predictions,batched_data),dim=-1).values
    true_implausibility = torch.sqrt(torch.ones(5)*2) / 2  # true result is 1/√2   
    assert results == approx(true_implausibility)



def test_implausibility_multivariate_distr_1d():
    data = MultivariateNormal(torch.tensor([0.0]), torch.eye(1))
    prediction = MultitaskMultivariateNormal(torch.tensor([[1.0]]), torch.eye(1))
    result = implausibility_multivariate_distr(prediction, data)
    true_implausibility_sq = 0.5  # true result is 1/2 (as this gives imp squared)
    assert result == approx(true_implausibility_sq) 

def test_implausibility_multivariate_distr_3d():
    data = MultivariateNormal(torch.tensor([0.0,0.0,0.0]), torch.eye(3))
    prediction = MultitaskMultivariateNormal(torch.tensor([[1.0,0.0,0.0]]), torch.eye(3))
    result = implausibility_multivariate_distr(prediction, data)
    true_implausibility_sq = 0.5  # true result is 1/2 (as this gives imp squared)
    assert result == approx(true_implausibility_sq) 


def test_implausibility_multivariate_distr_2d_corr():
    delta = 1e-5  # P.S.D correction
    data = MultivariateNormal(torch.tensor([0.0,0.0]), torch.ones(2,2)+torch.eye(2)*delta)
    prediction = MultitaskMultivariateNormal(torch.tensor([[1.0,1.0]]), torch.eye(2)*delta)
    result = implausibility_multivariate_distr(prediction, data)
    true_implausibility_sq = 1.0  # true result is 1 
    assert result == approx(true_implausibility_sq,rel=delta) ## result is only accurate for P.S.D correction. in reality for X=Y this is exact as perfect correlated variables are PSD but singular 
