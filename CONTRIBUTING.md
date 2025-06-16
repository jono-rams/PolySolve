# Contributing to polysolve

First off, thank you for considering contributing! We welcome any contributions, from fixing a typo to implementing a new feature.

## Table of Contents

* [Reporting Bugs](#reporting-bugs)
* [Suggesting Enhancements](#suggesting-enhancements)
* [Setting Up the Development Environment](#setting-up-the-development-environment)
* [Running Tests](#running-tests)
* [Pull Request Process](#pull-request-process)

## Reporting Bugs

If you find a bug, please open an issue on our Gitea issue tracker. Please include as many details as possible, such as your OS, Python version, steps to reproduce, and any error messages.

## Suggesting Enhancements

If you have an idea for a new feature or an improvement, please open an issue to discuss it. This allows us to coordinate efforts and ensure the proposed change aligns with the project's goals.

## Setting Up the Development Environment

1.  **Fork the repository** on Gitea.

2.  **Clone your fork** locally:
    ```bash
    git clone [https://gitea.example.com/YourUsername/PolySolve.git](https://gitea.example.com/YourUsername/PolySolve.git)
    cd PolySolve
    ```

3.  **Create and activate a virtual environment:**
    ```bash
    # For Unix/macOS
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    python -m venv .venv
    .venv\Scripts\activate
    ```

4.  **Install the project in editable mode with development dependencies.** This command installs `polysolve` itself, plus `pytest` for testing.
    ```bash
    pip install -e '.[dev]'
    ```

5.  If you need to work on the CUDA-specific features and test them, install the appropriate CuPy extra alongside the `dev` dependencies:
    ```bash
    # For a development setup with CUDA 12.x
    pip install -e '.[dev,cuda12]'

    # For a development setup with CUDA 11.x
    pip install -e '.[dev,cuda11]'
    ```

## Running Tests

We use `pytest` for automated testing. After setting up the development environment, you can run the full test suite with a single command from the root of the repository:

```bash
pytest
```

This will automatically discover and run all tests located in the `tests/` directory.

All tests should pass before you submit your changes. If you are adding a new feature or fixing a bug, please add a corresponding test case to ensure the code is working correctly and to prevent future regressions.

## Pull Request Process

1.  Create a new branch for your feature or bugfix from the `main` branch:
    ```bash
    git checkout -b your-feature-name
    ```
2.  Make your changes to the code in the `src/` directory.
3.  Add or update tests in the `tests/` directory to cover your changes.
4.  Run the test suite to ensure everything passes:
    ```bash
    pytest
    ```
5.  Commit your changes with a clear and descriptive commit message.
6.  Push your branch to your fork on Gitea.
7.  Open a pull request to the `main` branch of the original `PolySolve` repository. Please provide a clear title and description for your pull request.

Once you submit your pull request, our automated CI tests will run. We will review your contribution and provide feedback as soon as possible. Thank you for your contribution!
