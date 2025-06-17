# Contributing to polysolve

First off, thank you for considering contributing! We welcome any contributions, from fixing a typo to implementing a new feature.

## Table of Contents

* [Reporting Bugs](#reporting-bugs)
* [Suggesting Enhancements](#suggesting-enhancements)
* [Setting Up the Development Environment](#setting-up-the-development-environment)
* [Running Tests](#running-tests)
* [Our CI Process & The Gitea Bridge](#our-ci-process--the-gitea-bridge)
* [Pull Request Process](#pull-request-process)

## Reporting Bugs

If you find a bug, please open an issue on our Gitea issue tracker. Please include as many details as possible, such as your OS, Python version, steps to reproduce, and any error messages.  
[Report a Bug](https://github.com/jono-rams/PolySolve/issues/new?assignees=&labels=bug&template=bug_report.md&title=)

## Suggesting Enhancements

If you have an idea for a new feature or an improvement, please open an issue to discuss it. This allows us to coordinate efforts and ensure the proposed change aligns with the project's goals.  
[Suggest an Enhancement](https://github.com/jono-rams/PolySolve/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=)

## Setting Up the Development Environment

1.  **Fork the repository** on Gitea.

2.  **Clone your fork** locally:
    ```bash
    git clone git clone https://github.com/YourUsername/PolySolve.git
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
    ```

## Running Tests

We use `pytest` for automated testing. After setting up the development environment, you can run the full test suite with a single command from the root of the repository:

```bash
pytest
```

This will automatically discover and run all tests located in the `tests/` directory. All tests should pass before you submit your changes.  
If you are adding a new feature or fixing a bug, please add a corresponding test case to ensure the code is working correctly and to prevent future regressions.

## Our CI Process & The Gitea Bridge

To ensure that all contributions are consistent and stable, our test suite is executed automatically in a controlled environment. Here’s how it works:
1. Our canonical source of truth and CI/CD runners are managed on our private Gitea instance.
2. When you open a Pull Request on GitHub, our "Gitea Bridge" bot automatically mirrors your changes to a corresponding PR on our Gitea instance.
3. The tests are run using Gitea Actions within our specific, reproducible environment.
4. The results (success or failure) are then reported back to your GitHub Pull Request via a status check named "Gitea CI Bridge".

Your pull request must pass all these checks before it can be merged.

### Reference Environment Specification

You can replicate our CI environment exactly to minimize "it works on my machine" issues.
* **Base OS:** Ubuntu 24.04
* **CUDA Toolkit:** 12.5.1
* **Base Docker Image:** `nvidia/cuda:12.5.1-devel-ubuntu24.04`
* **Node.js Version:** 20.x
* **Python Versions Tested:** 3.8, 3.10, 3.12
* **CI Docker Image:** You can pull the exact image used by our runners from Docker Hub:

```bash
docker pull c1ph3rd3v/gitea-runner-cuda-node:12.5.1-ubuntu24.04
```

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
6.  Push your branch to your fork on GitHub.
7.  Open a **Pull Request** to the `main` branch of the `jono-rams/PolySolve` repository on **GitHub**. Please provide a clear title and description for your pull request.

Once you submit your pull request, our automated CI tests will run. We will review your contribution and provide feedback as soon as possible. Thank you for your contribution!
