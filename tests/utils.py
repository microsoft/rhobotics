import os
from functools import wraps

import pytest
import torch

# Test device configuration
DEVICE = os.environ.get("RHO_TEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def require_cuda(func):
    """Decorator to skip tests if CUDA is not available"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return func(*args, **kwargs)

    return wrapper


def require_package(package_name):
    """Decorator to skip tests if a package is not available"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                __import__(package_name)
            except ImportError:
                pytest.skip(f"Package {package_name} not available")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def assert_tensors_equal(tensor1, tensor2, rtol=1e-5, atol=1e-8):
    """Assert that two tensors are approximately equal"""
    if tensor1.shape != tensor2.shape:
        raise AssertionError(f"Shapes don't match: {tensor1.shape} vs {tensor2.shape}")

    if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        max_diff = torch.max(torch.abs(tensor1 - tensor2))
        raise AssertionError(f"Tensors not equal. Max difference: {max_diff}")
