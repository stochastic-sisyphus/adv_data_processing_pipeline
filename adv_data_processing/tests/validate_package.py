import os
from pathlib import Path

def validate_package_structure():
    root = Path(__file__).parent.parent
    
    # Required files
    required_files = [
        "setup.py",
        "README.md",
        "requirements.txt",
        "MANIFEST.in",
        "adv_data_processing/model_evaluation.py",
    ]
    
    # Check each required file
    missing_files = []
    for file in required_files:
        if not (root / file).exists():
            missing_files.append(file)
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required files: {', '.join(missing_files)}"
        )
    
    print("Package structure validation passed!")

if __name__ == "__main__":
    validate_package_structure() 