# TensorFlow Environment Installation

This document outlines the steps to install Miniforge, create a Conda environment, and set up the necessary dependencies to run the project.

## Prerequisites

Ensure that you have [Homebrew](https://brew.sh/) installed on your system.

## Installation Steps

### 1. Install Miniforge

Open your terminal and run the following command to install Miniforge using Homebrew:
```bash
brew install --cask miniforge
```

### 2. Create a Conda Environment

After installing Miniforge, create a new Conda environment named `tensorflow_env`:
```bash
conda create --name tensorflow_env python=3.9
```

### 3. Activate the Environment

Activate the environment you just created:
```bash
conda activate tensorflow_env
```

### 4. Install Dependencies

Ensure you are in the directory where your `requirements.txt` file is located, then run the following command to install all dependencies:
```bash
pip install -r requirements.txt
```

### 5. Run the Project

Once all dependencies are installed, you can run your main script:
```bash
python main.py
```

## Notes

- If you encounter compatibility issues, ensure that the library versions in `requirements.txt` are compatible with your version of Python and TensorFlow.
- To deactivate the Conda environment, use the following command:
```bash
conda deactivate
```
