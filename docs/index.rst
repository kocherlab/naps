|conda| |travis ci| |Documentation| |PyPI Upload| |Conda Upload| |LICENSE| |

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


.. image:: _static/logo.png
   :height: 2.2em
   :align: left

###############################
NAPS (NAPS is ArUco Plus SLEAP)
###############################


NAPS (NAPS is ArUco Plus SLEAP), is a tool for researchers with two goals: (1) to quantify animal behavior over a long timescale and high resolution, with minimal human bias, and (2) to track the behavior of individuals with a high level of identity-persistence. This could be of use to researchers studying social network analysis, animal communication, task specialization, or gene-by-environment interactions. By combining deep-learning based pose estimation software with easily read and minimally invasive fiducial markers ("tags"), we provide an easy-to-use solution for producing high-quality, high-dimensional behavioral data.

.. figure:: _static/example_tracking.gif
   :width: 600px
   :align: center
   :alt: Example usage of NAPS to track a colony of common eastern bumblebees.

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


.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: NAPS Documentation

   installation
   tutorials/index
   notebooks/index
   cli
   functions
   contact


