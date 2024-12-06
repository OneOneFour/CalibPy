import torch
import matplotlib.pyplot as plt
import pytest
from abracalibra.eki import EnsembleKalmanInversion
from abracalibra.forward_model import ForwardModel
from torch.distributions import Uniform, MultivariateNormal, Normal


@pytest.fixture
def linear_model():
    def f(x, m, c):
        return m * x + c

    return f


@pytest.fixture
def noise_linear_model_with_data(linear_model):
    m = 0.5
    c = 0.3
    x = torch.arange(0, 1 + 0.1, 0.1)
    y = linear_model(x, m, c)
    return x, y, (m, c)


@pytest.fixture
def noise_linear_model_eki_fixture(noise_linear_model_with_data):
    parameters = ["m", "c"]
    priors = {"m": Uniform(0, 1), "c": Uniform(0, 1)}
    x, y, theta = noise_linear_model_with_data
    noiselevel = 0.01
    noise = torch.randn_like(y) * noiselevel
    return x, y + noise, noiselevel, theta, parameters, priors


def test_make_eki(noise_linear_model_eki_fixture, linear_model):
    xdata, ydata, noiselevel, theta, parameters, priors = noise_linear_model_eki_fixture

    forward = ForwardModel(linear_model, vectorized=True, x=xdata)
    observables = MultivariateNormal(ydata, torch.eye(len(ydata)) * noiselevel**2)
    eki = EnsembleKalmanInversion(
        parameters, priors, observables, ensemble_size=20, dt=1, forward=forward
    )

    assert eki.current_time == 0
    assert eki.iteration == 0
    assert set(eki.parameter_names) == set(parameters)
    assert eki.number_of_observables == len(ydata)
    assert eki.number_of_parameters == 2
    assert eki.priors == priors

    assert eki.forward is not None


def test_eki_initial_ensemble(noise_linear_model_eki_fixture, linear_model):
    xdata, ydata, noiselevel, theta, parameters, priors = noise_linear_model_eki_fixture

    forward = ForwardModel(linear_model, vectorized=True)
    observables = MultivariateNormal(ydata, torch.eye(len(ydata)) * noiselevel**2)
    eki = EnsembleKalmanInversion(
        parameters, priors, observables, ensemble_size=20, dt=1, forward=forward
    )

    intial_ensemble = eki.generate_initial_ensemble()
    assert intial_ensemble.shape == (20, 2)

    assert (intial_ensemble[:, 0] >= 0).all()
    assert (intial_ensemble[:, 0] <= 1).all()
    assert (intial_ensemble[:, 1] >= 0).all()
    assert (intial_ensemble[:, 1] <= 1).all()


def test_eki_linear_problem(noise_linear_model_eki_fixture, linear_model):
    xdata, ydata, noiselevel, theta, parameters, priors = noise_linear_model_eki_fixture

    forward = ForwardModel(linear_model, vectorized=True, x=xdata)
    observables = MultivariateNormal(ydata, torch.eye(len(ydata)) * noiselevel**2)
    eki = EnsembleKalmanInversion(
        parameters, priors, observables, ensemble_size=100, dt=1, forward=forward
    )

    # I think this _should_ be exact for a linear model + gaussian after just 1 iteration
    mean_params = eki.calibrate(1)
    fig, ax = plt.subplots()
    ax.scatter(mean_params[0], mean_params[1], marker="x", color="black")
    for p in eki.parameters:
        ax.scatter(p[:, 0], p[:, 1])
    fig.savefig("tests/figures/test_eki_linear_problem.png")
    assert mean_params.shape == (2,)
    assert mean_params[0] == pytest.approx(theta[0], abs=0.05)
    assert mean_params[1] == pytest.approx(theta[1], abs=0.05)


def test_eki_non_linear_problem():
    x = torch.arange(0, 2 * torch.pi, 0.25)
    params = {"a": 1.0, "b": 0.2, "omega": 1.5, "phi": torch.pi * 0.12}
    priors = {
        "a": Normal(1, 0.5),
        "b": Normal(0, 0.5),
        "omega": Normal(1, 0.5),
        "phi": Uniform(-torch.pi / 2, torch.pi / 2),
    }
    noiselevel = 0.01

    def sin_model(x, a, b, omega, phi):
        return a * torch.sin(omega * x + phi) + b

    y = sin_model(x, **params) + torch.randn_like(x) * noiselevel
    forward = ForwardModel(sin_model, vectorized=True, x=x)
    observables = MultivariateNormal(y, torch.eye(len(y)) * noiselevel**2)
    eki = EnsembleKalmanInversion(
        list(params.keys()),
        priors,
        observables,
        ensemble_size=50,
        dt=1,
        forward=forward,
    )

    # I think this _should_ be exact for a linear model + gaussian after just 1 iteration
    mean_params = eki.calibrate(5)
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))

    for i, axs_col in enumerate(axs):
        for j, ax in enumerate(axs_col):
            if j == i:
                ax.hist(eki.parameters[-1][:, i], bins=20)
            else:
                ax.scatter(mean_params[i], mean_params[j], marker="x", color="black")
                ax.scatter(
                    list(params.values())[i],
                    list(params.values())[j],
                    marker="^",
                    color="red",
                )
                for p in eki.parameters:
                    ax.scatter(p[:, i], p[:, j], alpha=0.5, s=0.4)
                    ax.set_xlabel(list(params.keys())[i])
                    ax.set_ylabel(list(params.keys())[j])
    fig.tight_layout()
    fig.savefig("tests/figures/test_eki_nonlinear_problem_scatter.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    xplt = torch.linspace(0, 2 * torch.pi, 100)
    yplt = sin_model(xplt, *mean_params)
    ax.plot(xplt, yplt, label="mean")
    ax.scatter(x, y)
    fig.savefig("tests/figures/test_eki_nonlinear_problem_fit.png")

    assert mean_params.shape == (4,)
    assert mean_params[0] == pytest.approx(params["a"], rel=0.1)
    assert mean_params[1] == pytest.approx(params["b"], rel=0.1)
    assert mean_params[2] == pytest.approx(params["omega"], rel=0.1)
    assert mean_params[3] == pytest.approx(params["phi"], rel=0.1)
