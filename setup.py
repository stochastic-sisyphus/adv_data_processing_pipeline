from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="advanced-data-processing",
    version="0.2.7",  # Increment version
    author="Vanessa Beck",
    author_email="your.email@example.com",
    description="An advanced data processing pipeline for machine learning workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stochastic-sisyphus/adv_data_processing_pipeline",
    packages=find_packages(exclude=['tests*', 'test*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.8,<3.13',  # Update Python version constraint
    install_requires=[
        "pandas>=1.5.0",
        "dask[complete]>=2023.1.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.23.0",
        'beautifulsoup4>=4.11.0',
        'pyyaml>=6.0.0',
        'tenacity>=8.0.0',
        'joblib>=1.2.0',
        'tqdm>=4.65.0',
        'schema>=0.7.5',
        'sqlalchemy>=1.4.0',
        'boto3>=1.26.0',
        'requests>=2.28.0',
        'cerberus>=1.3.4',
        'psutil>=5.9.0',
        'torch>=2.1.0',
        'optuna>=3.3.0',
        'mlflow>=2.8.0'
    ],
)

