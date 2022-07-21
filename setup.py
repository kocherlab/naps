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

setup(name=kocher_tools.__title__,
      version=kocher_tools.__version__,
      project_urls={"Documentation": "https://github.com/kocherlab/naps",
                    "Code": "https://github.com/kocherlab/naps",
                    "Issue tracker": "https://github.com/kocherlab/naps/issues"},
      license=kocher_tools.__license__,
      url=kocher_tools.__url__,
      author=kocher_tools.__author__,
      author_email=kocher_tools.__email__,
      maintainer="Scott Wolf, Andrew Webb",
      maintainer_email="scott.w.wolf1@gmail.com, 19213578+aewebb80@users.noreply.github.com",
      description=kocher_tools.__summary__,
      long_description=readme,
      packages=['naps'],
      install_requires=requirements,
      scripts=tool_scripts,
      python_requires=">=3.7")
