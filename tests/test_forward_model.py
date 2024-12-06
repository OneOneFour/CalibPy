from abracalibra.forward_model import ForwardModel
import torch 
import pytest

@pytest.fixture 
def linear_model():
    def f(x,m,c):
        return m*x + c
    return f, ["x","m","c"]

@pytest.fixture
def sine_model():
    def f(x,a,b,omega,phi):
        return a*torch.sin(x*omega + phi) + b
    return f, ["x","a","b","omega","phi"]

@pytest.mark.parametrize("model", ["linear_model","sine_model"])    
@pytest.mark.parametrize("vectorized", [True, False])
def test_model_vector(model,vectorized,request):
    fn, param_names = request.getfixturevalue(model) 

    def vectorized_function(params):
        return fn(*params.unbind(-1))

    forward = ForwardModel(vectorized_function,vectorized=vectorized,params_as="vector")
    
    params = torch.zeros(1,len(param_names))
    res = forward(params)
    assert res == pytest.approx(0)

    params = torch.zeros(10,10,len(param_names))
    res = forward(params)
    assert not res.any()
    assert res.shape == (10,10)


        

@pytest.mark.parametrize("model", ["linear_model","sine_model"])    
@pytest.mark.parametrize("vectorized", [True, False])
def test_model_args(model,vectorized,request):
    fn, param_names = request.getfixturevalue(model)
    forward = ForwardModel(fn,vectorized=vectorized,params_as="args")
    params = torch.zeros(1,len(param_names))    
    res = forward(params)
    assert res == pytest.approx(0)

    params = torch.zeros(10,10,len(param_names))
    res = forward(params)
    assert not res.any()
    assert res.shape == (10,10)

@pytest.mark.parametrize("model", ["linear_model","sine_model"])    
@pytest.mark.parametrize("vectorized", [True, False])  
def test_model_kwargs(model,vectorized,request):
    fn, param_names = request.getfixturevalue(model)

    forward = ForwardModel(fn,vectorized=vectorized,params_as="kwargs")

    params = torch.zeros(1,len(param_names))    
    res = forward(params)
    assert res == pytest.approx(0)
    assert res.shape == (1,)

    params = torch.zeros(10,10,len(param_names))
    res = forward(params)
    assert not res.any()
    assert res.shape == (10,10)





@pytest.mark.parametrize("model", ["linear_model","sine_model"])
@pytest.mark.parametrize("vectorized", [True, False])
def test_model_kwargs_with_data(model,vectorized,request):
    fn, param_names = request.getfixturevalue(model)
    forward = ForwardModel(fn,vectorized=vectorized,params_as="kwargs")
    N_DATA = 100
    forward = forward.bind(x=torch.arange(N_DATA).double())

    params = torch.zeros(1,len(param_names)-1)
    res = forward(params)
    assert not res.any()
    assert res.shape == (1,N_DATA)


    params = torch.zeros(10,10,len(param_names)-1)
    res = forward(params)
    assert not res.any()
    assert res.shape == (10,10,N_DATA)