# Data clustering

CLI tool for data clustering structured as a package.

# Table of Contents

-   [Installation](#installation)
-   [Usage](#usage)
-   [Pre-commit Setup](#pre-commit-setup)
-   [License](#license)

---

# Installation

Follow these steps to install the project package.

### **1. Clone repository:**

Clone repository:

```git
git clone https://github.com/michalmoninec/data_clustering.git
```

Change directory to project:

```bash
cd data_clustering
```

### **2. Check that `Python` is installed:**

-   Python 3.12 or higher is required.
-   Ensure that Python is installed by running:

    On Windows:

    ```bash
    python --version
    ```

    On Linux, macOS:

    ```bash
    python3 --version
    ```

### **3. Create virtual enviroment and activate it:**

On Windows:

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

On Linux, macOS:

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

### **4. Upgrade pip and install package:**

On Windows:

```bash
python -m pip install --upgrade pip
```

On Linux, macOS:

```bash
python3 -m pip install --upgrade pip
```

**Install the package with all the dependencies:**

```bash
pip install .
```

---

## Usage

Usage description.

---

## Pre-commit Setup

Project uses pre-commit hooks, that checks linting, formatting, typing, etc.
Instructions for setting up pre-commit hooks are:

### **1. Install `pre-commit`:**

Install the `pre-commit` package:

```bash
pip install pre-commit
```

Install the hooks specified in
[`.pre-commit-config.yaml`](.pre-commit-config.yaml):

```bash
pre-commit install
```

Run locally:

```bash
pre-commit run --all-files
```

---

## License

This project is licensed under the MIT License - see
the [LICENSE](LICENSE) file for details.
