##############################
NAPS Installation Instructions
##############################

*************
Conda (Linux)
*************

To install ``naps-track`` in a clean environment with `SLEAP <https://sleap.ai/>`_, run the following:

.. code-block:: bash

    conda create -n naps naps-track -c kocherlab -c sleap -c nvidia -c conda-forge

***
pip
***

``naps-track`` may also be installed using pip:

.. code-block:: bash

    pip install naps-track

******
GitHub
******

If you are interested in following the development of ``naps-track``, you may want to install the latest development version of ``naps-track`` directly from the `GitHub repository <htto://github.com/kocherlab/naps-track>`_.


1. Ensure git is installed:

.. code-block:: bash

    sudo apt-get install git

2. Clone the repository:

.. code-block:: bash

    git clone https://github.com/talmolab/sleap && cd sleap

3. Install NAPS:

.. code-block:: bash

    pip install .


