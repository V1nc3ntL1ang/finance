from __future__ import annotations

import importlib.metadata
import sys


EXPECTED_PYTHON = (3, 11, 15)
EXPECTED_PACKAGES = {
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "scipy": "1.17.1",
    "scikit-learn": "1.9.0",
    "lightgbm": "4.6.0",
    "matplotlib": "3.10.9",
    "joblib": "1.5.3",
    "tqdm": "4.68.2",
    "Pillow": "12.2.0",
}


def main() -> None:
    actual_python = sys.version_info[:3]
    if actual_python != EXPECTED_PYTHON:
        expected = ".".join(str(part) for part in EXPECTED_PYTHON)
        actual = ".".join(str(part) for part in actual_python)
        raise RuntimeError(f"Python version mismatch: expected {expected}, got {actual}")

    mismatches: list[str] = []
    for package, expected_version in EXPECTED_PACKAGES.items():
        actual_version = importlib.metadata.version(package)
        if actual_version != expected_version:
            mismatches.append(f"{package}: expected {expected_version}, got {actual_version}")

    if mismatches:
        raise RuntimeError("Environment mismatch: " + "; ".join(mismatches))

    print("environment check passed", flush=True)


if __name__ == "__main__":
    main()
