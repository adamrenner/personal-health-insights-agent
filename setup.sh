#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Function to check Python version ---
check_python_version() {
    echo "Checking Python version..."
    # Get the Python version using the specified python3 interpreter
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d " " -f 2)
    # Extract major and minor version numbers
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d "." -f 1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d "." -f 2)

    # Check if the version is 3.11 or higher
    if ! { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; }; then
        echo "Error: Python 3.11 or higher is required. You have version $PYTHON_VERSION."
        echo "Please install a compatible Python version and ensure 'python3' points to it."
        exit 1
    fi
    echo "Python version $PYTHON_VERSION is compatible."
}

# --- Main setup function ---
setup_venv() {
    # Check Python version before proceeding
    check_python_version

    VENV_NAME=".venv"
    echo "Setting up Python virtual environment '$VENV_NAME'..."

    # Remove the existing virtual environment directory if it exists
    if [ -d "$VENV_NAME" ]; then
        echo "Removing existing '$VENV_NAME' directory..."
        rm -rf "$VENV_NAME"
    fi

    # Create a new virtual environment
    python3 -m venv "$VENV_NAME"
    echo "Virtual environment created."

    # Activate the virtual environment
    # Note: The activation is temporary for this script's execution.
    # The user will need to run 'source phia_env/bin/activate' manually.
    source "$VENV_NAME/bin/activate"
    echo "Virtual environment activated."

    # Upgrade pip to the latest version
    echo "Upgrading pip..."
    pip install --upgrade pip

    # Verify pip and python info for debugging
    echo "Using pip from: $(which pip)"
    echo "Python version: $(python --version)"

    # Install dependencies from requirements.txt
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt

    # Install the onetwo library from GitHub with --no-deps
    echo "Installing onetwo library..."
    pip install --no-deps git+https://github.com/google-deepmind/onetwo

    # Register the kernel with Jupyter
    echo "Registering Jupyter kernel..."
    python -m ipykernel install --user --name="$VENV_NAME" --display-name="PHIA Environment ($VENV_NAME)"

    # Deactivate the environment at the end of the script
    deactivate

    echo ""
    echo "----------------------------------------------------------------"
    echo "Setup complete! The '$VENV_NAME' environment is ready."
    echo "To activate it, run the following command in your terminal:"
    echo "source $VENV_NAME/bin/activate"
    echo "----------------------------------------------------------------"
}

# --- Execute the setup function ---
setup_venv
