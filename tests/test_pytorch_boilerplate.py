from pytorch_boilerplate import __version__


def test_version():
    assert __version__ == '0.1.0'

def test_cuda_available():
    import torch

    assert torch.cuda.is_available() == True
