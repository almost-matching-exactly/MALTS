import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymalts",
    version="0.1.0",
    author="Harsh Parikh",
    author_email="harsh.parikh@duke.edu",
    description="Causal Inference using Matching",
    long_description="We introduce a flexible framework that produces high-quality almost-exact matches for causal inference. Most prior work in matching uses ad-hoc distance metrics, often leading to poor quality matches, particularly when there are irrelevant covariates. In this work, we learn an interpretable distance metric for matching, which leads to substantially higher quality matches. The learned distance metric stretches the covariates according to their contribution to outcome prediction. The framework is flexible in that the user can choose the form of the distance metric and the type of optimization algorithm. Our ability to learn flexible distance metrics leads to matches that are interpretable and useful for the estimation of conditional average treatment effects.",
    long_description_content_type="text/markdown",
    url="https://github.com/almost-matching-exactly/MALTS",
    project_urls={
        "Bug Tracker": "https://github.com/almost-matching-exactly/MALTS/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",
)