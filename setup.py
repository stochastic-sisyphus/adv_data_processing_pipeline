from setuptools import setup, find_packages

setup(
    name="advanced-data-processing",
    version="0.2.0",
    packages=find_packages(exclude=['tests*', 'test*']),
    install_requires=[
        'dask>=2023.3.0',
        'pandas>=1.5.0',
        'numpy>=1.23.0',
        'scikit-learn>=1.0.0',
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
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-mock>=3.10.0',
            'pytest-asyncio>=0.21.0',
            'hypothesis>=6.75.3',
        ],
    },
    author="Vanessa Beck",
    description="An advanced data processing pipeline",
    python_requires='>=3.8',
)

