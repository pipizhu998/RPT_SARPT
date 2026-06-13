# Utility Unit Tests

These tests verify utility and experiment-infrastructure code using Python's
built-in `unittest` framework. They use temporary directories, synthetic arrays,
small tensors, and mocks; they do not download datasets or train models.
Temporary test artifacts are created under `testing/.tmp/`.

Run all tests from the repository root:

```powershell
python testing/run_tests.py
```

Individual test files can also be run directly from `testing`:

```powershell
cd testing
python test_analysis_utilities.py
```

Coverage includes:

- `test_experiment_config.py`: config loading, merging, includes, and validation.
- `test_core_utils.py`: seeds, devices, file output, progress, and evaluation.
- `test_evaluation_utilities.py`: output paths, cache keys, and recorders.
- `test_analysis_utilities.py`: result parsing, aggregation, and analysis helpers.

The tests expect all dependencies from `requirements.txt` to be installed.
