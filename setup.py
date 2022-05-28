import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="easysrv",
    author="Daniel Wiczew",
    author_email="daniel.wiczew@gmail.com",
    description="Easy to use State Reversible Vampnet with fit, transform and fittransform methods",
    keywords="machine learning, molecular dynamics, tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DanielWicz/easysrv",
    project_urls={
        "Documentation": "https://github.com/DanielWicz/easysrv",
        "Bug Reports": "https://github.com/DanielWicz/easysrv/issues",
        "Source Code": "https://github.com/DanielWicz/easysrv",
        # 'Funding': '',
        # 'Say Thanks!': '',
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        # see https://pypi.org/classifiers/
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["numpy", "tensorflow", "scipy"],
    extras_require={
        "dev": ["check-manifest"],
    },
)
