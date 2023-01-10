.. _tutorial:

=========
Tutorials
=========

The tutorials here give a basic idea of the workflow of NAPS along with some example analysis and quality control code.

----------------
Basic NAPS Usage
----------------

.. _preparing-dataset:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Preparing a dataset for NAPS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Running NAPS on the command line
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most direct usage of NAPS is through the CLI. After installing NAPS, :ref:`naps-track<cli-naps-track>` covers the most basic usage. Check out :ref:`naps-track<cli-naps-track>` for the full documentation of the interface including default values.

.. code-block:: text

    naps-track [-h] --slp-path SLP_PATH [--h5-path H5_PATH] --video-path
        VIDEO_PATH --tag-node TAG_NODE_NAME --start-frame START_FRAME
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

    naps-track --slp-path example.slp --video-path example.mp4 --tag-node-name tag --start-frame 0 --end-frame 1200 --aruco-marker-set DICT_5X5_50 --output-path example-naps.slp --aruco-error-correction-rate 0.6 --aruco-adaptive-thresh-constant 7 --aruco-adaptive-thresh-win-size-max 23 --aruco-adaptive-thresh-win-size-step 10 --aruco-adaptive-thresh-win-size-min 3 --half-rolling-window-size 20

The resulting file, `example-naps.slp`, is in the SLEAP Project file format and can be opened directly with SLEAP.

.. _post-tracking:

----------
After NAPS
----------
If we want to post process data from NAPS, it's easy to follow directly with SLEAP tutorials. See `sleap.ai/tutorials/tutorial <https://sleap.ai/tutorials/tutorial>`_.
We can also remove unassigned tracks by opening the SLEAP GUI > Custom Instance Delete > Delete all instances with no track identity. The remaining tracks with have been assigned to ArUco tags and the track names will expose the n

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example: Importing HDF5 files into Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The resulting HDF5 can be read into Python, as shown in the SLEAP tutorial, as follows:

.. code-block:: python

    import h5py
    import numpy as np

    filename = "example-naps.analysis.h5"

    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        track_names = f["track_names"]
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

    print("===track names===")
    for i, name in enumerate(track_names):
        print(f"{i}: {name}")
    print()

    print("===nodes===")
    for i, name in enumerate(node_names):
        print(f"{i}: {name}")
    print()

