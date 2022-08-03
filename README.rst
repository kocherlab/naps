.. Template taken from https://github.com/talmolab/sleap
|conda| |conda| |travis ci| |Documentation| |PyPI Upload| |Conda Upload| |LICENSE| |

.. |travis ci| image::
   https://app.travis-ci.com/kocherlab/naps.svg?branch=main
   :target: https://app.travis-ci.com/kocherlab/naps
   :alt: Continuous integration status

.. |Documentation| image::
   https://readthedocs.org/projects/naps/badge/?version=latest
   :target: https://naps.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |conda| image::
   https://anaconda.org/kocherlab/naps-track/badges/version.svg
   :target: https://anaconda.org/kocherlab/naps-track

.. |Conda Upload| image::
   https://github.com/kocherlab/naps/actions/workflows/upload_conda.yml/badge.svg
   :target: https://github.com/kocherlab/naps/actions/workflows/upload_conda.yml

.. |PyPI Upload| image::
   https://github.com/kocherlab/naps/actions/workflows/python-publish.yml/badge.svg
   :target: https://github.com/kocherlab/naps/actions/workflows/python-publish.yml

.. |LICENSE| image::
   https://anaconda.org/kocherlab/naps-track/badges/license.svg
   :target: https://github.com/kocherlab/naps/blob/main/LICENSE.md

NAPS
====

NAPS documentation can be found at `naps.readthedocs.io <https://naps.readthedocs.io/en/latest/>`_.

###############################
NAPS (NAPS is ArUco Plus SLEAP)
###############################


NAPS (NAPS is ArUco Plus SLEAP), is a tool for researchers with two goals: (1) to quantify animal behavior over a long timescale and high resolution, with minimal human bias, and (2) to track the behavior of individuals with a high level of identity-persistence. This could be of use to researchers studying social network analysis, animal communication, task specialization, or gene-by-environment interactions. By combining deep-learning based pose estimation software with easily read and minimally invasive fiducial markers ("tags"), we provide an easy-to-use solution for producing high-quality, high-dimensional behavioral data.

.. figure:: https://naps.readthedocs.io/en/latest/_static/example_tracking.gif
   :width: 600px
   :align: center
   :alt: Example usage of NAPS to track a colony of common eastern bumblebees.

Getting NAPS
--------------

Easy install
^^^^^^^^^^^^^
`conda` **(Windows/Linux/GPU)**:

.. code-block:: bash

    conda create -n naps naps-track -c kocherlab -c sleap -c nvidia -c conda-forge


`pip` **(any OS)**:

.. code-block:: bash

    pip install naps-track


-----------
References
-----------

If you use NAPS in your research, please cite:

   \S. W. Wolf, D. M. Ruttenberg*, D. Y. Knapp*, A. W. Webb, J. W. Shaevitz, S. D. Kocher. `Hybrid tracking to capture dynamic social networks <https://naps.rtfd.io/>`__. *In Prep*, n.d.

~~~~~~~~~~~~~~~~~~~~~~
BibTeX:
~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

   @UNPUBLISHED{Wolf_undated,
      author = {Wolf, Scott W and Ruttenberg, Dee M and Knapp, Daniel Y and Webb,
               Andrew E and Shaevitz, Joshua W and Kocher, Sarah D},
      title = {Hybrid tracking to capture dynamic social networks},
      year = {n.d.}
   }


------------
Contributors
------------

* **Scott Wolf**, Lewis-Sigler Institute, Princeton University
* **Dee Ruttenberg**, Physics, Princeton University
* **Daniel Knapp**, Physics, Princeton University
* **Andrew Webb**, Ecology and Evolutionary Biology and Lewis-Sigler Institute, Princeton University
* **Joshua Shaevitz**, Physics and Lewis-Sigler Institute, Princeton University
* **Sarah Kocher**, Ecology and Evolutionary Biology and Lewis-Sigler Institute, Princeton University

-----------
License
-----------

NAPS is licensed under the MIT license. See the `LICENSE <https://github.com/kocherlab/naps/blob/main/LICENSE.md>`_ file for details.

