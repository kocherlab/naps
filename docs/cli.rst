.. _cli:

Command line interfaces
--------------------------

.. _cli-naps-track:

.. attention::

   Many of the parameters of the command line interfaces are simply passed to OpenCV's ArUco library. For more information on these parameters, please refer to the `ArUco documentation <https://docs.opencv.org/4.6.0/d9/d6a/group__aruco.html>`_.

.. argparse::
   :module: naps.naps_track
   :func: build_parser
   :prog: naps-track

