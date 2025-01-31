[![Python CI](https://github.com/jan-meissner/AnonGraph/actions/workflows/CI.yml/badge.svg)](https://github.com/jan-meissner/AnonGraph/actions/workflows/CI.yml)

# AnonymiGraph

Anonymization and Privacy Preserving Algorithms for Graphs in Python.

Check out the github repo at: https://github.com/jan-meissner/AnonGraph

## Table of Contents

- [Codespaces: Run the Code Without Installations](#codespaces-run-the-code-without-installations)
- [Usage](#usage)
- [Local Installation](#manual-installation)
  - [From Git Repository](#from-git-repository)
  - [Local Installation](#local-installation)

## Codespaces: Run the Code Without Installations

No need to worry about any installation.

1. Navigate to the GitHub repository (https://github.com/jan-meissner/AnonGraph).
2. Click on the 'Code' button and select 'Codespaces' and then click on 'Create codespace on main'.
3. Wait for ~10 minutes for the environment to setup and then you are ready to go. (see [Usage](#usage) for usage)

## Usage

Execute the notebooks found in the notebooks folder. If you are using Codespaces select the default starred kernel for the Jupyter Notebooks.

## Local Installation

### From Git Repository

To install the package from the Git repository, run the following command:

```bash
pip install git+https://github.com/jan-meissner/AnonGraph.git
```

### Local Installation

For local development, you can install the project by following these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/username/repository.git
   ```
2. Navigate to the project directory:
   ```bash
   cd repository
   ```
3. Install the package:
   ```bash
   pip install -e .[full,test,dev,torch]
   ```
4. Set up pre-commits:
   ```bash
   pre-commit install
   ```
5. Run pytest to check if your setup worked
   ```bash
   pytest .
   ```
