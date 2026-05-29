"""setup.py — Install HQCT pipeline as a pip package."""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="hqct",
    version="1.0.0",
    description=(
        "Hybrid Quantum-Classical Transformer for clinical risk stratification "
        "(CKD + Framingham Heart Study)"
    ),
    author="HQCT Research Team",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "hqct-ckd=main:main",
            "hqct-fhs=main_fhs:main",
            "hqct-ablation=ablation_study:main",
            "hqct-tables=report.tables:generate_all_tables",
            "hqct-figures=utils.publication_plots:generate_all_figures",
            "hqct-sanity=scripts.sanity_check:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
