import io
import naps
from setuptools import setup

with io.open("README.rst", "rt", encoding="utf8") as f:
    readme = f.read()

# List of required non-standard python libraries
requirements = ['pandas',
                'tox']

# Executable scripts in the package
tool_scripts = ['naps/matching.py']

setup(name=naps.__title__,
      version=naps.__version__,
      project_urls={"Documentation": naps.__docs__,
                    "Code": naps.__code__,
                    "Issue tracker": naps.__issue__},
      license=naps.__license__,
      url=naps.__url__,
      author=naps.__author__,
      author_email=naps.__email__,
      maintainer=naps.__maintainer__,
      maintainer_email=naps.__maintainer_email__,
      description=naps.__summary__,
      long_description=readme,
      packages=['naps'],
      install_requires=requirements,
      scripts=tool_scripts,
      python_requires=">=3.7")
