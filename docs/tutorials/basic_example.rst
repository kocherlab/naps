.. _basic-tutorial:

Basic NAPS Usage
----------------

.. _preparing-dataset:

Preparing a dataset for NAPS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What you need to proceed:

#. A `.slp` file with putative tracks from SLEAP (and ideally a `.h5` analysis file corresponding to the SLEAP file.)
#. The video used to generate the tracks
#. Knowledge of the tag node in your SLEAP skeleton

We start by preparing a dataset for NAPS. To run NAPS, you will first need to have a SLEAP file with putative tracks assigned by one of the trackers in SLEAP. We strongly recommend going through the tutorials on `sleap.ai <https://sleap.ai>`_ to learn how to do this.

For the purpose of this tutorial, we will use the toy data provided in `tests/data/` on the `NAPS GitHub repository <https://github.com/kocherlab/naps>`_. The data includes a `.slp` file with putative tracks from SLEAP, a `.h5` analysis file corresponding to the SLEAP file, and the corresponding video file.

.. note::

    While NAPS will allow you to directly input the results of `sleap-track`, we strongly recommend you also convert this to an analysis h5 using `sleap-convert` if you will be troubleshooting your workflow. Directly reading in the locations from the h5 drastically increases speed.

.. _tracking:

.. note::

    While this tutorial is more complete, we also provide a simple example of running the workflow on Google Colaboratory which can be found in the notebooks section :ref:`Notebooks<notebooks>`

Running NAPS on the command line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most direct usage of NAPS is through the CLI. After installing NAPS, :ref:`naps-track<cli-naps-track>` covers the most basic usage. Check out :ref:`naps-track<cli-naps-track>` for the full documentation of the interface including default values.

.. code-block:: text

    naps-track [-h] --slp-path SLP_PATH [--h5-path H5_PATH] --video-path
        VIDEO_PATH --tag-node TAG_NODE --start-frame START_FRAME
        --end-frame END_FRAME
        [--half-rolling-window-size HALF_ROLLING_WINDOW_SIZE]
        --aruco-marker-set ARUCO_MARKER_SET
        [--aruco-crop-size ARUCO_CROP_SIZE]
        [--aruco-adaptive-thresh-win-size-min ADAPTIVETHRESHWINSIZEMIN]
        [--aruco-adaptive-thresh-win-size-max ADAPTIVETHRESHWINSIZEMAX]
        [--aruco-adaptive-thresh-win-size-step ADAPTIVETHRESHWINSIZESTEP]
        [--aruco-adaptive-thresh-constant ADAPTIVETHRESHCONSTANT]
        [--aruco-perspective-rm-ignored-margin PERSPECTIVEREMOVEIGNOREDMARGINPERCELL]
        [--aruco-error-correction-rate ERRORCORRECTIONRATE]
        [--output-path OUTPUT_PATH] [--threads THREADS]

.. important::

    While the defaults arguments work for the example data, nearly all usage of NAPS beyond the example dataset will require fine tuning of parameters.

.. code-block:: text

    naps-track --slp-path tests/data/example.slp --h5-path tests/data/example.analysis.h5 --video-path tests/data/example.mp4 --tag-node 0 --start-frame 0 --end-frame 1203 --aruco-marker-set DICT_4X4_100 --output-path tests/data/example_output.analysis.h5

The resulting file, `example_output.analysis.h5`, is the same form as SLEAP analysis HDF5s.

.. _post-tracking:

After NAPS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The resulting HDF5 can be imported back into SLEAP using SLEAP's import function -- treating it as a SLEAP analysis HDF5.

Example of importing the HDF5 into Python
*****************************************

The resulting HDF5 can also be read into Python, as shown in the SLEAP tutorial, as follows:

.. code-block:: python

    import h5py
    import numpy as np

    filename = "tests/data/example_output.analysis.h5"

    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]

    print("===filename===")
    print(filename)
    print()

    print("===HDF5 datasets===")
    print(dset_names)
    print()

    print("===locations data shape===")
    print(locations.shape)
    print()

    print("===nodes===")
    for i, name in enumerate(node_names):
        print(f"{i}: {name}")
    print()

