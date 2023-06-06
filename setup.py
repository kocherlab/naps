import io
import os

from setuptools import setup

import naps

with io.open("README.rst", "rt", encoding="utf8") as f:
    readme = f.read()

# Read requirements.txt
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + "/requirements.txt"
requirements = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        requirements = f.read().splitlines()

# Executable scripts in the package
tool_scripts = ["naps/naps_track.py", "naps/naps_plot.py", "naps/naps_interactions.py"]

setup(
    name=naps.__name__,
    version=naps.__version__,
    project_urls={
        "Documentation": naps.__docs__,
        "Code": naps.__code__,
        "Issue tracker": naps.__issue__,
    },
    license=naps.__license__,
    url=naps.__url__,
    description=naps.__summary__,
    long_description_content_type="text/x-rst",
    long_description=readme,
    packages=["naps", "naps.utils"],
    install_requires=requirements,
    scripts=tool_scripts,
    entry_points={
        "console_scripts": [
            "naps-track=naps.naps_track:main",
            "naps-plot=naps.naps_plot:main",
            "naps-interactions=naps.naps_interactions:main",
        ],
    },
    python_requires=">=3.7",
)
