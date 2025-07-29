#!/bin/bash
# Development environment setup script for Gaudi 3 Scale

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Gaudi 3 Scale development environment...${NC}"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if [ "$(echo "$PYTHON_VERSION >= 3.10" | bc)" -eq 0 ]; then
    echo -e "${RED}Error: Python 3.10+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in development mode
echo -e "${YELLOW}Installing package in development mode...${NC}"
pip install -e .

# Install pre-commit hooks
echo -e "${YELLOW}Installing pre-commit hooks...${NC}"
pre-commit install

# Create necessary directories
echo -e "${YELLOW}Creating project directories...${NC}"
mkdir -p data models logs configs

# Run initial tests
echo -e "${YELLOW}Running initial tests...${NC}"
pytest tests/ -v

echo -e "${GREEN}🎉 Development environment setup complete!${NC}"
echo -e "${YELLOW}To activate the environment, run: source venv/bin/activate${NC}"
echo -e "${YELLOW}To run tests: pytest${NC}"
echo -e "${YELLOW}To run linting: pre-commit run --all-files${NC}"