from functools import partial
import torch
import pytest
from torch.distributions import Uniform
from abracalibra.approximatebayesiancomputation import ApproximateBayesianComputation
from abracalibra.forward_model import ForwardModel
import matplotlib.pyplot as plt


@pytest.fixture
def linear_model():
    def f(x, m, c):
        return m * x + c

    return f


@pytest.fixture
def linear_model_with_data(linear_model):
    m = 0.5
    c = 0.3
    x = torch.arange(0, 1 + 0.1, 0.1)
    y = linear_model(x, m, c)
    return x, y


@pytest.fixture
def noise_linear_model_with_data(linear_model_with_data):
    x, y = linear_model_with_data
    noiselevel = 0.03
    noise = torch.randn_like(y) * noiselevel
    return x, y + noise, noiselevel


@pytest.fixture
def noise_linear_model_abc_fixture(noise_linear_model_with_data):
    parameters = ["m", "c"]
    priors = {"m": Uniform(0, 1), "c": Uniform(0, 1)}

    trainx, trainy, noise = noise_linear_model_with_data

    def summary(y, true_y):
        return torch.sqrt(torch.sum((y - true_y) ** 2))

    tolerance = 0.1
    return parameters, priors, trainx, trainy, summary, tolerance, noise


def test_create_abc(noise_linear_model_abc_fixture):
    ## Linear model y = mx + c
    parameters, priors, trainx, trainy, summary, tolerance, noise = (
        noise_linear_model_abc_fixture
    )

    abc = ApproximateBayesianComputation(parameters, priors, trainy, summary, tolerance)
    assert abc.parameter_names == parameters
    assert abc.priors == priors
    assert (abc.observables == trainy).all()
    assert abc.tolerance == tolerance
    assert abc.forward is None
    assert abc.accepted_samples is None

    assert abc.summary(trainy) == pytest.approx(0.0)


def test_abc_sample_prior(noise_linear_model_abc_fixture):
    parameters, priors, trainx, trainy, summary, tolerance, noise = (
        noise_linear_model_abc_fixture
    )
    abc = ApproximateBayesianComputation(parameters, priors, trainy, summary, tolerance)

    samples = abc.sample_prior(100)
    assert samples.shape == (100, 2)
    assert samples[:, 0].min() >= 0
    assert samples[:, 0].max() <= 1
    assert samples[:, 1].min() >= 0
    assert samples[:, 1].max() <= 1

    samples = abc.sample_prior(100, labeled=True)
    assert isinstance(samples, dict)
    assert samples.keys() == {"m", "c"}
    assert samples["m"].min() >= 0
    assert samples["m"].max() <= 1
    assert samples["c"].min() >= 0
    assert samples["c"].max() <= 1

    batch_samples = abc.sample_prior(torch.Size([100, 100]))
    assert batch_samples.shape == (100, 100, 2)
    assert batch_samples[:, 0].min() >= 0
    assert batch_samples[:, 0].max() <= 1
    assert batch_samples[:, 1].min() >= 0
    assert batch_samples[:, 1].max() <= 1


def test_calibrate_abc_linear(noise_linear_model_with_data, linear_model):
    parameters = ["m", "c"]
    priors = {"m": Uniform(0, 1), "c": Uniform(0, 1)}

    trainx, trainy, noise = noise_linear_model_with_data

    plt.scatter(trainx, trainy)

    def summary(y, true_y):
        return torch.sqrt(torch.sum((y - true_y) ** 2 / (noise**2), dim=-1))

    def forward(m, c):
        return linear_model(trainx, m, c)

    tolerance = 5
    fwd_model = ForwardModel(forward)
    abc = ApproximateBayesianComputation(
        parameters, priors, trainy, summary, tolerance, forward=fwd_model
    )
    accepted_samples = abc.sample_posterior(100)

    x = torch.linspace(0, 1, 10)
    y = linear_model(x, 0.5, 0.3)
    mean_sample = accepted_samples.mean(dim=0)
    y_mean = linear_model(x, *mean_sample)
    y_sample = ForwardModel(forward)(accepted_samples)

    plt.title(f"ABC Calibration, num calls:{fwd_model.call_count}")
    plt.plot(x, y)
    plt.plot(x, y_mean, linestyle="--")
    for i in range(y_sample.shape[0]):
        plt.plot(trainx, y_sample[i], alpha=0.1)
    plt.savefig("tests/figures/test_calibrate_abc.png")

    assert accepted_samples.shape == (100, 2)
    assert accepted_samples[:, 0].mean() == pytest.approx(0.5, abs=0.1)
    assert accepted_samples[:, 1].mean() == pytest.approx(0.3, abs=0.1)


def test_calibrate_abc_linear_batch_params(noise_linear_model_with_data, linear_model):
    parameters = ["m", "c"]
    priors = {"m": Uniform(0, 1), "c": Uniform(0, 1)}

    trainx, trainy, noise = noise_linear_model_with_data

    plt.scatter(trainx, trainy)

    def summary(y, true_y):
        return torch.sqrt(torch.sum((y - true_y) ** 2 / (noise**2), dim=-1))

    def forward(m, c):
        return linear_model(trainx, m, c)

    tolerance = 5
    fwd_model = ForwardModel(forward)
    abc = ApproximateBayesianComputation(
        parameters, priors, trainy, summary, tolerance, forward=fwd_model
    )
    accepted_samples = abc.sample_posterior(
        100, batch_params=torch.Size([10]), verbose=True
    )

    x = torch.linspace(0, 1, 10)
    y = linear_model(x, 0.5, 0.3)
    mean_sample = accepted_samples.mean(dim=0)
    y_mean = linear_model(x, *mean_sample)
    y_sample = ForwardModel(forward)(accepted_samples)

    plt.title(f"ABC Calibration, num calls:{fwd_model.call_count}")
    plt.plot(x, y)
    plt.plot(x, y_mean, linestyle="--")
    for i in range(y_sample.shape[0]):
        plt.plot(trainx, y_sample[i], alpha=0.1)
    plt.savefig("tests/figures/test_calibrate_abc_batch_params.png")

    assert accepted_samples.shape == (100, 2)
    assert accepted_samples[:, 0].mean() == pytest.approx(0.5, abs=0.1)
    assert accepted_samples[:, 1].mean() == pytest.approx(0.3, abs=0.1)


def test_calibrate_abc_linear_vectorize(noise_linear_model_with_data, linear_model):
    parameters = ["m", "c"]
    priors = {"m": Uniform(0, 1), "c": Uniform(0, 1)}

    trainx, trainy, noise = noise_linear_model_with_data

    plt.scatter(trainx, trainy)

    def summary(y, true_y):
        return torch.sqrt(torch.sum((y - true_y) ** 2 / (noise**2), dim=-1))

    def forward(m, c):
        m = m.unsqueeze(-1)
        c = c.unsqueeze(-1)
        xdata = trainx.unsqueeze(0)
        return linear_model(xdata, m, c)

    tolerance = 5
    fwd_model = ForwardModel(forward, vectorized=True)
    abc = ApproximateBayesianComputation(
        parameters, priors, trainy, summary, tolerance, forward=fwd_model
    )
    accepted_samples = abc.sample_posterior(
        1000, batch_params=torch.Size([10]), verbose=True
    )

    x = torch.linspace(0, 1, 10)
    y = linear_model(x, 0.5, 0.3)
    mean_sample = accepted_samples.mean(dim=0)
    y_mean = linear_model(x, *mean_sample)
    y_sample = ForwardModel(forward, vectorized=True)(accepted_samples)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"ABC Calibration, num calls:{fwd_model.call_count}")
    ax1.plot(x, y)
    ax1.plot(x, y_mean, linestyle="--")
    for i in range(y_sample.shape[0]):
        ax1.plot(trainx, y_sample[i], alpha=0.005)
    ax2.hist2d(accepted_samples[:, 0], accepted_samples[:, 1], bins=(10, 10))
    ax2.set_xlabel("m")
    ax2.set_ylabel("c")
    fig.savefig("tests/figures/test_calibrate_abc_vectorize.png")

    assert accepted_samples.shape == (1000, 2)
    assert accepted_samples[:, 0].mean() == pytest.approx(0.5, abs=0.1)
    assert accepted_samples[:, 1].mean() == pytest.approx(0.3, abs=0.1)


def test_calibrate_abc_nonlinear_vectorize():
    parameters = ["a", "b", "omega", "phi"]
    priors = {
        "a": Uniform(0.5, 1.5),
        "b": Uniform(0, 0.5),
        "omega": Uniform(1, 2),
        "phi": Uniform(-torch.pi / 2, torch.pi / 2),
    }
    noise = 0.3

    def sin_model(x, a, b, omega, phi):
        return a * torch.sin(x * omega + phi) + b

    PARAMS = {"a": 1.0, "b": 0.2, "omega": 1.5, "phi": torch.pi * 0.12}

    trainx = torch.arange(0, 2 * torch.pi, 0.5)
    trainy = sin_model(trainx, **PARAMS) + torch.randn_like(trainx) * noise

    def summary(y, true_y):
        return torch.sqrt(
            torch.sum((y - true_y) ** 2 / (noise**2), dim=-1) / true_y.shape[-1]
        )

    tolerance = 1
    fwd_model = ForwardModel(sin_model, vectorized=True, x=trainx)
    abc = ApproximateBayesianComputation(
        parameters, priors, trainy, summary, tolerance, forward=fwd_model
    )
    accepted_samples = abc.sample_posterior(
        1000, batch_params=torch.Size([1000]), verbose=True
    )

    x = torch.linspace(0, 2 * torch.pi, 1000)
    y = sin_model(x, **PARAMS)

    y_sample = ForwardModel(sin_model, vectorized=True, x=x)(accepted_samples)

    plt.title(f"ABC Calibration, num calls:{fwd_model.call_count*1000}")
    plt.scatter(trainx, trainy)

    # plt.plot(x,y_mean,linestyle='--')
    for i in range(y_sample.shape[0]):
        plt.plot(x, y_sample[i], alpha=0.05)
    plt.plot(x, y)
    plt.savefig("tests/figures/test_calibrate_sine_abc_vectorize.png")

    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(4):
        for j in range(4):
            if i == j:
                axs[i, j].hist(accepted_samples[:, i], bins=10)
            else:
                axs[i, j].hist2d(
                    accepted_samples[:, i], accepted_samples[:, j], bins=(10, 10)
                )
                # axs[i,j].set_xlim(priors[parameters[i]].low,priors[parameters[i]].high)
                # axs[i,j].set_ylim(priors[parameters[i]].low,priors[parameters[i]].high)
                axs[i, j].set_xlabel(parameters[i])
                axs[i, j].set_ylabel(parameters[j])
    fig.tight_layout()
    fig.savefig("tests/figures/test_calibrate_sine_abc_vectorize_cross_plot.png")

    assert accepted_samples.shape == (1000, 4)
    assert accepted_samples[:, 0].mean() == pytest.approx(PARAMS["a"], abs=noise)
    assert accepted_samples[:, 1].mean() == pytest.approx(PARAMS["b"], abs=noise)
    assert accepted_samples[:, 2].mean() == pytest.approx(PARAMS["omega"], abs=noise)
    assert accepted_samples[:, 3].mean() == pytest.approx(PARAMS["phi"], abs=noise)
