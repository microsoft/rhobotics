# ALKU Tests

This directory contains the test suite for the ALKU robotics package. The tests are organized following the same structure as the main package and use pytest as the testing framework.

## Test Structure

```
tests/
├── __init__.py                     # Test package initialization
├── conftest.py                     # Pytest configuration and global fixtures
├── utils.py                        # Test utilities and helper functions
├── test_available.py              # Basic import and availability tests
├── fixtures/                      # Test fixtures
│   ├── datasets.py               # Dataset-related fixtures
│   └── policies.py               # Policy-related fixtures
├── common/                        # Tests for rho.common module
│   ├── test_env.py               # Environment tests
│   └── test_wandb_logging.py     # W&B logging tests
├── datasets/                      # Tests for rho.datasets module
│   └── test_dataset.py           # Dataset configuration and loading tests
├── policies/                      # Tests for rho.policies module
│   └── test_policies.py          # Policy configuration and creation tests
├── training/                      # Tests for rho.training module
│   ├── test_train.py             # Training configuration tests
│   └── test_train_utils.py       # Training utility tests
└── utils/                         # Tests for rho.utils module
    └── test_utils.py              # Environment factory tests
```

## Running Tests

### Prerequisites

Install the test dependencies:

```bash
pip install -e ".[test]"
```

### Basic Test Execution

Run all tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=rho
```

Run specific test files:
```bash
pytest tests/common/test_env.py
pytest tests/policies/test_policies.py
```

Run tests by marker:
```bash
pytest -m "not slow"      # Skip slow tests
pytest -m "unit"          # Run only unit tests
pytest -m "integration"   # Run only integration tests
```

### Test Categories

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Slow tests**: Tests that take longer to run (marked with `@pytest.mark.slow`)
- **GPU tests**: Tests that require CUDA (marked with `@pytest.mark.gpu`)
- **Environment tests**: Tests that require specific environments (marked with `@pytest.mark.env`)

### Environment Variables

- `ALKU_TEST_DEVICE`: Set device for testing ("cuda" or "cpu", default: auto-detect)
- `WANDB_MODE=offline`: Run W&B tests in offline mode

### Mock Dependencies

Many tests use mocks to avoid requiring actual hardware or external services:

- **gym_pusht**: Environment tests mock the PushT environment when not available
- **wandb**: W&B logging tests mock the wandb API
- **torch.load/save**: Checkpoint tests mock file I/O operations

## Writing New Tests

### Test Organization

1. **Follow the package structure**: Create test files that mirror the package structure
2. **Use descriptive names**: Test functions should clearly describe what they test
3. **Group related tests**: Use classes to group related test functions
4. **Use fixtures**: Leverage pytest fixtures for common setup

### Example Test Structure

```python
def test_component_initialization():
    """Test basic initialization"""
    pass

def test_component_custom_config():
    """Test with custom configuration"""
    pass

def test_component_error_handling():
    """Test error cases"""
    pass

@pytest.mark.slow
def test_component_integration():
    """Test integration with other components"""
    pass
```

### Common Patterns

1. **Configuration Testing**:
   ```python
   def test_config_initialization():
       config = SomeConfig()
       assert config.param == expected_value
   ```

2. **Mock Testing**:
   ```python
   @patch('module.function')
   def test_with_mock(mock_function):
       mock_function.return_value = expected_value
       result = function_under_test()
       assert result == expected_result
   ```

3. **Error Testing**:
   ```python
   def test_error_case():
       with pytest.raises(ValueError, match="expected error message"):
           function_that_should_fail()
   ```

## Test Fixtures

Common fixtures are defined in `fixtures/` directory:

- `dummy_dataset_config`: Basic dataset configuration for testing
- `dummy_policy_config`: Basic policy configuration for testing
- `pusht_feature_dict`: PushT-specific feature dictionary
- `sample_checkpoint_path`: Temporary checkpoint file

## Coverage

The test suite aims for high coverage of the ALKU package. Coverage reports are generated in:

- Terminal: Shows coverage summary after test run
- HTML: `htmlcov/index.html` - Detailed coverage report
- XML: `coverage.xml` - For CI/CD integration

## Continuous Integration

Tests are designed to run in CI environments:

- **Headless mode**: Tests work without display/GUI
- **CPU fallback**: GPU tests gracefully fallback to CPU
- **Mock dependencies**: External dependencies are mocked when not available
- **Deterministic**: Tests use fixed seeds for reproducibility

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure the package is installed in development mode (`pip install -e .`)
2. **GPU tests failing**: Set `ALKU_TEST_DEVICE=cpu` to force CPU testing
3. **Missing dependencies**: Install test dependencies (`pip install -e ".[test]"`)
4. **Timeout issues**: Use `-m "not slow"` to skip long-running tests

### Debug Mode

Run tests with verbose output and no coverage for debugging:
```bash
pytest -v -s --no-cov tests/path/to/test.py::test_function
```
