from setuptools import setup, find_packages

setup(
    name="tr_dynamic_preprocess",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jpype1",
        "gensim",
        "fasttext-wheel",
        "pandas",
        "numpy",
        "tqdm",
        "scikit-learn",
        "matplotlib"
    ],
    author="Ferat Kılın",
    url="https://github.com/Feratkln/tr-dynamic-preprocess",
    description="Turkish Dynamic Preprocessing Library using Zemberek and FastText",
    long_description=open("README.md", encoding="utf-8").read() if open("README.md") else "Turkish Dynamic Preprocessing",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
