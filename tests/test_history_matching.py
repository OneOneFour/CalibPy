import pytest
import torch
import matplotlib.pyplot as plt
from torch.distributions import Uniform, Normal, MultivariateNormal
from abracalibra.summary import batched_implausibility_univariate_distr,implausibility_univariate_distr
from abracalibra.history_matching import HistoryMatchingBase, HistoryMatchingSingleTaskGP,HistoryMatchingSingleTaskGPLastWaveOnly,HistoryMatchingMultiTask
from abracalibra.forward_model import ForwardModel


@pytest.fixture
def linear_model():
    def f(x, m, c):
        return m * x + c

    return f


@pytest.fixture
def linear_model_with_data(linear_model):
    m = 0.5
    c = 0.3
    x = torch.arange(0, 1 + 0.1, 0.05)
    y = linear_model(x, m, c)
    return x, y, m,c


@pytest.fixture
def noise_linear_model_with_data(linear_model_with_data):
    x, y,m,c = linear_model_with_data
    noiselevel = 0.03
    noise = torch.randn_like(y) * noiselevel
    true_params = (m,c)
    return x, y + noise, noiselevel,true_params


@pytest.fixture
def noise_linear_model_hm_fixture(noise_linear_model_with_data):
    parameters = ["m", "c"]
    priors = {"m": Uniform(0, 1), "c": Uniform(0, 1)}

    trainx, trainy, noise,true_params = noise_linear_model_with_data

    return parameters, priors, trainx, trainy, noise,true_params


def test_cannot_create_history_matching_base(
    noise_linear_model_hm_fixture, linear_model
):
    parameters, priors, trainx, trainy, noise, true_params= noise_linear_model_hm_fixture
    fwd_model = ForwardModel(linear_model, vectorized=True, x=trainx)

    def summary(y, true_y):
        true_y_dist = Normal(true_y, torch.ones_like(true_y) * noise)
        return torch.max(implausibility_univariate_distr(y, true_y_dist), dim=-1)

    with pytest.raises(TypeError):
        HistoryMatchingBase(
            parameters, priors, trainy, summary, ensemble_size=10, forward=fwd_model
        )  # pylint: disable=abstract-class-instantiated


def test_create_history_matching_single_task_gp(
    noise_linear_model_hm_fixture, linear_model
):
    parameters, priors, trainx, trainy, noise, true_params = noise_linear_model_hm_fixture
    fwd_model = ForwardModel(linear_model, vectorized=True, x=trainx)

    def summary(y, true_y):
        true_y_dist = Normal(true_y, torch.ones_like(true_y) * noise)
        return torch.max(implausibility_univariate_distr(y, true_y_dist), dim=-1).values

    hm = HistoryMatchingSingleTaskGP(
        parameters, priors, trainy, summary, ensemble_size=10, forward=fwd_model
    )
    assert set(hm.parameter_names) == set(parameters)
    assert hm.priors == priors
    assert (hm.observables == trainy).all()

    assert hm.summary(Normal(trainy, torch.ones_like(trainy) * noise)) == pytest.approx(
        0.0
    )
    assert hm.implausibility_cutoff == 3.0
    assert hm.ensemble_size == 10
    assert hm.forward == fwd_model
    assert len(hm.emulators) == 0
    assert len(hm.params) == 0
    assert len(hm.results) == 0
    assert hm.current_wave == 0

    # Should not be converged initially even if threshold > 0
    assert not hm.check_if_converged(0.01)


def test_history_matching_single_task_gp_sample_prior(
    noise_linear_model_hm_fixture, linear_model
):
    parameters, priors, trainx, trainy, noise, true_params = noise_linear_model_hm_fixture
    fwd_model = ForwardModel(linear_model, vectorized=True, x=trainx)

    def summary(y, true_y):
        true_y_dist = Normal(true_y, torch.ones_like(true_y) * noise)
        return torch.max(implausibility_univariate_distr(y, true_y_dist), dim=-1).values

    hm = HistoryMatchingSingleTaskGP(
        parameters, priors, trainy, summary, ensemble_size=10, forward=fwd_model
    )
    samples = hm.sample_prior(100)
    assert samples.shape == (100, 2)
    assert samples[:, 0].min() >= 0
    assert samples[:, 0].max() <= 1
    assert samples[:, 1].min() >= 0
    assert samples[:, 1].max() <= 1

    samples = hm.sample_prior(100, labeled=True)
    assert samples.keys() == set(parameters)
    for p in parameters:
        assert samples[p].min() >= 0
        assert samples[p].max() <= 1


def test_history_matching_generate_initial_ensemble(
    noise_linear_model_hm_fixture, linear_model
):
    parameters, priors, trainx, trainy, noise,true_params = noise_linear_model_hm_fixture
    fwd_model = ForwardModel(linear_model, vectorized=True, x=trainx)

    def summary(y, true_y):
        true_y_dist = Normal(true_y, torch.ones_like(true_y) * noise)
        return torch.max(implausibility_univariate_distr(y, true_y_dist), dim=-1).values

    hm = HistoryMatchingSingleTaskGP(
        parameters, priors, trainy, summary, ensemble_size=10, forward=fwd_model
    )

    ensemble = hm.generate_initial_ensemble()

    assert ensemble.shape == (10, 2)
    assert ensemble[:, 0].min() >= 0
    assert ensemble[:, 0].max() <= 1
    assert ensemble[:, 1].min() >= 0
    assert ensemble[:, 1].max() <= 1


def test_history_matching_gp_initial_nroy(noise_linear_model_hm_fixture, linear_model):
    parameters, priors, trainx, trainy, noise, true_params= noise_linear_model_hm_fixture
    fwd_model = ForwardModel(linear_model, vectorized=True, x=trainx)

    def summary(y, true_y):
        true_y_dist = Normal(true_y, torch.ones_like(true_y) * noise)
        return torch.max(implausibility_univariate_distr(y, true_y_dist), dim=-1).values

    hm = HistoryMatchingSingleTaskGP(
        parameters, priors, trainy, summary, ensemble_size=10, forward=fwd_model
    )

    inbounds = hm.nroy(torch.tensor([0.5, 0.5]))
    outofbounds = hm.nroy(torch.tensor([1.5, 0.5]))
    assert inbounds.all()
    assert not outofbounds.all()

    multi_tensor = torch.tensor([[0.5, 0.5], [1.5, 0.5], [1.5, 0.5], [1.5, 1.5]])
    assert multi_tensor.shape == (4, 2)
    nroy_test = hm.nroy(multi_tensor)
    assert (nroy_test == torch.tensor([True, False, False, False])).all()
    assert nroy_test.shape == (4,)

    nroy_test = hm.nroy(multi_tensor.reshape(2, 2, 2))
    assert nroy_test.shape == (2, 2)
    assert (nroy_test == torch.tensor([[True, False], [False, False]])).all()


def test_history_matching_gp_calibrate_single_summary_linear(noise_linear_model_hm_fixture,linear_model):
    parameters, priors, trainx, trainy, noise, true_params = noise_linear_model_hm_fixture

    fwd_model = ForwardModel(lambda x,m,c: linear_model(x.mean(),m,c), vectorized=True, x=trainx)
    NWAVES = 2
    observables = Normal(trainy.mean(), torch.sqrt(trainy.var() / len(trainy)))

    hm = HistoryMatchingSingleTaskGP(
        parameters,
        priors,
        observables,
        implausibility_univariate_distr,
        ensemble_size=10,
        forward=fwd_model
    )
    hm.calibrate(NWAVES,verbose=True)


    fig, ax = plt.subplots()
    nroy, (m, c) = hm.nroy_over_support(return_coords=True)
    ax.pcolormesh(c, m, nroy)
    ax.set_xlabel("c")
    ax.set_ylabel("m")
    ax.scatter(true_params[1], true_params[0], color="red",label="True")
    ax.legend()
    fig.savefig("tests/figures/test_history_matching_gp_calibrate_single_summary_linear.png")

    assert hm.params.shape == (NWAVES, 10, 2)
    assert hm.results.shape == (NWAVES, 10, 1)
    assert len(hm.emulators) == NWAVES
    assert hm.current_wave == NWAVES
    is_true_in_nroy = hm.nroy(torch.tensor([[0.5,0.3]]))
    assert is_true_in_nroy


def test_history_matching_gp_calibrate_multi_summary_linear(noise_linear_model_hm_fixture, linear_model):
    parameters, priors, trainx, trainy, noise, true_params= noise_linear_model_hm_fixture

    def mean_linear_model(x,y, m, c):
        mean = linear_model(x.mean(), m, c).squeeze()
        res = torch.sqrt(
            torch.mean((y - (m*x + c ))**2,dim=-1)
        )
        return torch.stack([mean, res], dim=-1)

    fwd_model = ForwardModel(mean_linear_model, vectorized=True,x=trainx,y=trainy)
    NWAVES = 3
    observables = MultivariateNormal(
        loc=torch.tensor([trainy.mean(), 0.0]),
        covariance_matrix=torch.diag(
            torch.tensor([trainy.var() / len(trainy), noise * noise])
        ),
    )

    def summary(y, true_y):
        return torch.max(batched_implausibility_univariate_distr(y, true_y), dim=-1).values

    hm = HistoryMatchingSingleTaskGP(
        parameters,
        priors,
        observables,
        summary,
        ensemble_size=20,
        forward=fwd_model,
    )

    hm.calibrate(NWAVES,verbose=True)

    fig, ax = plt.subplots()
    nroy, (m, c) = hm.nroy_over_support(return_coords=True)
    ax.pcolormesh(c, m, nroy)
    ax.set_xlabel("c")
    ax.set_ylabel("m")
    ax.scatter(true_params[1], true_params[0], color="red",label="True")
    ax.legend()
    fig.savefig("tests/figures/test_history_matching_gp_calibrate_multi_summary_linear.png")

    assert hm.params.shape == (NWAVES, 20, 2)
    assert hm.results.shape == (NWAVES, 20, 2)
    assert len(hm.emulators) == NWAVES
    assert hm.current_wave == NWAVES
    is_true_in_nroy = hm.nroy(torch.tensor([[0.5,0.3]]))
    assert is_true_in_nroy

def test_history_matching_gp_calibrate_multi_summary_linear_lwo(noise_linear_model_hm_fixture, linear_model):
    parameters, priors, trainx, trainy, noise, true_params= noise_linear_model_hm_fixture

    def mean_linear_model(x,y, m, c):
        mean = linear_model(x.mean(), m, c).squeeze()
        res = torch.sqrt(
            torch.mean((y - (m*x + c ))**2,dim=-1)
        )
        return torch.stack([mean, res], dim=-1)

    fwd_model = ForwardModel(mean_linear_model, vectorized=True,x=trainx,y=trainy)
    NWAVES = 3
    observables = MultivariateNormal(
        loc=torch.tensor([trainy.mean(), 0.0]),
        covariance_matrix=torch.diag(
            torch.tensor([trainy.var() / len(trainy), noise * noise])
        ),
    )

    def summary(y, true_y):
        return torch.max(batched_implausibility_univariate_distr(y, true_y), dim=-1).values

    hm = HistoryMatchingSingleTaskGPLastWaveOnly(
        parameters,
        priors,
        observables,
        summary,
        ensemble_size=20,
        forward=fwd_model,
    )

    hm.calibrate(NWAVES,verbose=True)

    fig, ax = plt.subplots()
    nroy, (m, c) = hm.nroy_over_support(return_coords=True)
    ax.pcolormesh(c, m, nroy)
    ax.set_xlabel("c")
    ax.set_ylabel("m")
    ax.scatter(true_params[1], true_params[0], color="red",label="True")
    for wave in range(hm.current_wave):
        ax.scatter(hm.params[wave,:,1], hm.params[wave,:,0], label=f"Wave {wave}",marker='x')
    ax.legend()
    fig.savefig("tests/figures/test_history_matching_gp_calibrate_multi_summary_linear_lwo.png")

    assert hm.params.shape == (NWAVES, 20, 2)
    assert hm.results.shape == (NWAVES, 20, 2)
    assert len(hm.emulators) == NWAVES
    assert hm.current_wave == NWAVES
    is_true_in_nroy = hm.nroy(torch.tensor([[0.5,0.3]]))
    assert is_true_in_nroy


def test_history_matching_gp_calibrate_nonlinear():
    x = torch.arange(0,2*torch.pi,0.5)   
    params = {
        "a":1.0,
        "b": 0.2,
        "omega":1.5,
        "phi":torch.pi*0.12
    }
    priors = {
        "a": Uniform(0.5,1.5),
        "b": Uniform(0,0.5),
        "omega": Uniform(1,2),
        "phi":Uniform(-torch.pi/2,torch.pi/2)
    }
    noiselevel = 0.01
    def sin_model(x,a,b,omega,phi):
        return a*torch.sin(omega*x + phi) + b
    
    y = sin_model(x,**params) + torch.randn_like(x)*noiselevel
    forward = ForwardModel(sin_model,vectorized=True,x=x) 
    observables = MultivariateNormal(y,torch.eye(len(y))*noiselevel**2)

    def summary(y, true_y):
        return torch.max(implausibility_univariate_distr(y, true_y), dim=-1).values
    NWAVES = 1
    hm = HistoryMatchingMultiTask(
        list(params.keys()),
        priors,
        observables,
        summary,
        ensemble_size=40,
        forward=forward,
        implausibility_cutoff=3,
    )
    hm.calibrate(NWAVES,verbose=True)
    em = hm.emulators[0]
    a = params["a"] * torch.ones(100)
    b = params["b"] * torch.ones(100)
    omega = torch.linspace(1,2,100)
    phi = params["phi"] * torch.ones(100)
    em.eval()
    pred_y = em(torch.stack([a,b,omega,phi],dim=-1))
    imp = summary(pred_y,observables)
    with torch.no_grad():
        lower,upper = pred_y.confidence_region()
        plt.plot(omega,pred_y.mean[:,10])
        plt.scatter(params["omega"],y[10])
        plt.fill_between(omega,lower[:,10],upper[:,10],alpha=0.5)
        plt.savefig("tests/figures/test_history_matching_gp_calibrate_nonlinear_plot1.png")
        plt.close()
        plt.axhline(3,color='red')
        plt.plot(omega,imp)
        plt.savefig("tests/figures/test_history_matching_gp_calibrate_nonlinear_plot2.png")



    fig, ax = plt.subplots(nrows=4,ncols=4,figsize=(15,15))
    nroy, axes = hm.nroy_over_support(num_points=30,return_coords=True)
    is_in_nroy = hm.nroy(torch.tensor([p for p in params.values()]))
    for i,axs_col in enumerate(ax):
        for j,ax in enumerate(axs_col):

            if i > j:
                nroy_dim = nroy.sum(dim=[d for d in range(nroy.dim()) if d not in [i,j]])/(30*30)
                X,Y = torch.meshgrid(axes[i],axes[j],indexing='xy')
                ax.pcolormesh(X,Y ,nroy_dim,shading='nearest')
                ax.set_xlabel(list(params.keys())[i])
                ax.set_ylabel(list(params.keys())[j])
                ax.scatter(params[list(params.keys())[i]],params[list(params.keys())[j]],marker='^',color='red')
            else:
                ax.axis("off")

    fig.tight_layout()   
    fig.savefig("tests/figures/test_history_matching_gp_calibrate_multi_summary_nonlinear.png")

    assert is_in_nroy