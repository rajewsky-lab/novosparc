from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

try:
    from novosparc import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ''
        
setup(
    name="novotest",
    version="0.2.2.2",
    author=__author__,
    author_email=__email__,
    description="De novo spatial reconstruction of single-cell gene expression.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rajewsky-lab/novosparc",
    license='MIT',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X"
    ]
)
