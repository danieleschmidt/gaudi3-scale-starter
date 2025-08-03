#!/usr/bin/env python3
"""Update version across project files."""

import sys
import re
from pathlib import Path


def update_pyproject_toml(version: str) -> None:
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("pyproject.toml not found")
        return
    
    content = pyproject_path.read_text()
    updated_content = re.sub(
        r'version = "[^"]*"',
        f'version = "{version}"',
        content
    )
    pyproject_path.write_text(updated_content)
    print(f"Updated pyproject.toml version to {version}")


def update_init_py(version: str) -> None:
    """Update version in __init__.py."""
    init_path = Path("src/gaudi3_scale/__init__.py")
    if not init_path.exists():
        print("__init__.py not found")
        return
    
    content = init_path.read_text()
    updated_content = re.sub(
        r'__version__ = "[^"]*"',
        f'__version__ = "{version}"',
        content
    )
    init_path.write_text(updated_content)
    print(f"Updated __init__.py version to {version}")


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <version>")
        sys.exit(1)
    
    version = sys.argv[1]
    print(f"Updating version to: {version}")
    
    update_pyproject_toml(version)
    update_init_py(version)
    
    print("Version update complete")


if __name__ == "__main__":
    main()