from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="odvm",
    version="0.1.0",
    author="Omnia Ayman",
    author_email="omnia18ayman@gmail.com",
    description="Automated analysis and machine learning for big data using Dask",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/omnia197/odvm",
    packages=find_packages(include=["odvm", "odvm.*"]),
    python_requires=">=3.8",
    install_requires=[
        "dask>=2023.0.0",
        "distributed>=2023.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "xgboost>=1.5.0",
        "dask-ml>=1.9.0",
    ],
    extras_require={
        "complete": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "fastparquet>=0.8.0",
            "dask-sql>=2023.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": ["odvm-run=odvm.cli:main"],
    },
    package_data={
        "odvm": ["config/*.json", "models/*.pkl"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Data Scientists",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8+",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "machine-learning",
        "big-data",
        "automation",
        "dask",
    ],
)