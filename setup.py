# Author: Kenta Nakamura <c60evaporator@gmail.com>
# Copyright (c) 2020-2021 Kenta Nakamura
# License: BSD 3 clause

from setuptools import setup
import segmentation_tool


def _requires_from_file(filename):
    return open(filename).read().splitlines()

DESCRIPTION = "segmentation_tool: Tool for struggle to segmentation task"
NAME = 'segmentation_tool'
AUTHOR = 'Masashi Ueda'
AUTHOR_EMAIL = 'masashi620@gmail.com'
URL = 'https://github.com/masashi2ueda/segmentation_tool'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/masashi2ueda/segmentation_tool'
VERSION = segmentation_tool.__version__
PYTHON_REQUIRES = ">=3.9"

EXTRAS_REQUIRE = {
}

PACKAGES = [
    'segmentation_tool'
]

CLASSIFIERS = [
]

setup(name=NAME,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    python_requires=PYTHON_REQUIRES,
    install_requires=_requires_from_file('requirements.txt'),
    extras_require=EXTRAS_REQUIRE,
    packages=PACKAGES,
    classifiers=CLASSIFIERS
    )