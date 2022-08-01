.. _installation:

##############################
NAPS Installation Instructions
##############################

*************
Conda (Linux)
*************

.. attention::

    This is the recommended installation method. While it requires setting up conda prior to use, it provides a more stable and reliable environment.


To install NAPS in a clean environment with `SLEAP <https://sleap.ai/>`_, run the following:

.. code-block:: bash

    conda create -n naps naps-track -c kocherlab -c sleap -c nvidia -c conda-forge

.. hint::

    If you haven't installed Conda before, you must install a version of Conda (Anaconda, Miniconda, etc) before running the above snippet. We recommend Miniconda as a lightweight version of Anaconda.

    Miniconda can be installed as follows:

    1. Download the appropriate `Miniconda installer <https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links>`_.
    2. Follow the installer instructions.

***
pip
***

NAPS may also be installed using pip:

.. code-block:: bash

    pip install naps-track

.. note::

    Requires Python 3.7.



******
GitHub
******

If you are interested in following the development of NAPS, you may want to install the latest development version of NAPS directly from the `GitHub repository <htto://github.com/kocherlab/naps-track>`_.


To do that, do the following:

1. Ensure git is installed:

.. code-block:: bash

    git --version

.. note::

    While many systems have git preinstalled, you may need to install git. This can be done by following the instructions on `git-scm.com/downloads <https://git-scm.com/downloads>`_.

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/kocherlab/naps.git && cd naps

3. Install NAPS:

.. code-block:: bash

    pip install .


