from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tabular-preprocessor",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive library for preprocessing tabular data in ML pipelines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tabular-preprocessor",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/tabular-preprocessor/issues",
        "Documentation": "https://github.com/yourusername/tabular-preprocessor#readme",
        "Source Code": "https://github.com/yourusername/tabular-preprocessor",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "imbalanced-learn>=0.9.0",
        "ctgan>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    keywords=[
        "machine learning",
        "preprocessing",
        "tabular data",
        "feature engineering",
        "data augmentation",
        "SMOTE",
        "CTGAN",
        "medical imaging",
        "radiomics",
    ],
)
