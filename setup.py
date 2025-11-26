from pathlib import Path
from setuptools import setup, find_packages


setup(
    name="textvec",
    version="3.0",
    description="Supervised text features extraction",
    long_description=Path("README.md").read_text(encoding="utf8"),
    long_description_content_type="text/markdown",
    url="https://github.com/textvec/textvec",
    author="Alex Zverianskii",
    author_email="alex@zverianskii.com",
    license="MIT",
    keywords="text nlp vectorization scikit-learn",
    packages=find_packages(exclude=['examples']),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23,<2.0",
        "scipy>=1.10,<1.13",
        "scikit-learn>=1.3,<1.6",
        "gensim>=4.3,<5.0",
        "tomli>=2.0.1; python_version < '3.11'",
    ],
    extras_require={
        "dev": ["pytest>=7.4,<9.0"],
        "test": ["pytest>=7.4,<9.0"],
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
)
