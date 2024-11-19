from setuptools import setup, find_packages
import os

def check_dependencies():
    with open('requirements.txt') as f:
        required = f.read().splitlines()
    return required

def validate_package():
    # Check for required files
    required_files = ['README.md', 'requirements.txt', 'MANIFEST.in']
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file {file} not found")

if __name__ == '__main__':
    validate_package()
    
    setup(
        name='adv_data_processing',
        version='0.1.0',
        packages=find_packages(),
        install_requires=check_dependencies(),
        python_requires='>=3.8',
        test_suite='tests',
    ) 