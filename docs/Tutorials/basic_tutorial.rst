.. _basic-tutorial:

Basic NAPS Tutorial
--------------------------

.. _preparing-dataset:

Preparing a dataset for NAPS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What you need to proceed:

#. `.slp` file with putative tracks from SLEAP
#. The video used to generate the tracks
#. Knowledge of the tag node in your SLEAP skeleton

We start by preparing a dataset for NAPS. For the purpose of this tutorial, we will use the toy data provided in `tests/data/`.

To run NAPS, you will first need to have a SLEAP file with putative tracks assigned by one of the trackers in SLEAP. We strongly recommend going through the tutorials on `sleap.ai <https://sleap.ai>`_ to learn how to do this.

While NAPS will allow you to directly input the results of `sleap-track`, we strongly recommend you also convert this to an analysis h5 using `sleap-convert` if you will be troubleshooting your workflow. Directly reading in the locations from the h5 drastically increases speed.

Running NAPS on the command line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most direct usage of NAPS is through the CLI. After installing NAPS, :ref:`naps-track<cli-naps-track>` covers the most basic usage. Check out :ref:`naps-track<cli-naps-track>` for the full documentation of the interface including default values.

.. code-block:: text

    naps-track [-h] [--slp-path SLP_PATH]
                    [--video-path VIDEO_PATH]
                    --tag-node TAG_NODE
                    [--start-frame START_FRAME]
                    [--end-frame END_FRAME]
                    [--half-rolling-window-size HALF_ROLLING_WINDOW_SIZE]
                    [--aruco-marker-set ARUCO_MARKER_SET]
                    [--aruco-crop-size ARUCO_CROP_SIZE]
                    [--aruco-adaptive-thresh-win-size-min ADAPTIVETHRESHWINSIZEMIN]
                    [--aruco-adaptive-thresh-win-size-max ADAPTIVETHRESHWINSIZEMAX]
                    [--aruco-adaptive-thresh-win-size-step ADAPTIVETHRESHWINSIZESTEP]
                    [--aruco-adaptive-thresh-constant ADAPTIVETHRESHCONSTANT]
                    [--aruco-perspective-rm-ignored-margin PERSPECTIVEREMOVEIGNOREDMARGINPERCELL]
                    [--aruco-error-correction-rate ERRORCORRECTIONRATE]
                    [--output-path OUTPUT_PATH]
                    [--threads THREADS]



As an example, we can run the following command to run NAPS on the toy data provided in `tests/data/`:

.. code-block:: text

    naps-track --slp-path tests/data/example.slp --video-path tests/data/example.mp4 --tag-node 0 --output-path tests/data/example_output.analysis.h5

The resulting file, `example_output.analysis.h5`, is the same form as SLEAP analysis HDF5s and can be read, as shown in the SLEAP tutorial, as follows:

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

