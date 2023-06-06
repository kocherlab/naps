#!/usr/bin/env python
import argparse
import logging
import os
import time

from naps.utils.tracking import TrackingArray

logger = logging.getLogger("NAPS Logger")


def plotParser():
    """
    Plot Parser

    Assign the parameters for plot assignment

    Parameters
    ----------
    sys.argv : list
            Parameters from command line

    Raises
    ------
    IOError
            If the specified files do not exist
    """

    def confirmFile():
        """Custom action to confirm file exists"""

        class customAction(argparse.Action):
            def __call__(self, parser, args, value, option_string=None):
                if not os.path.isfile(value):
                    raise IOError("%s not found" % value)
                setattr(args, self.dest, value)

        return customAction

    plot_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input files
    plot_parser.add_argument(
        "--sleap-file",
        help="Defines the input SLEAP file",
        type=str,
        action=confirmFile(),
        required=True,
    )
    plot_parser.add_argument(
        "--yaml-model",
        help="Defines the YAML model file",
        type=str,
        action=confirmFile(),
        required=True,
    )

    plot_parser.add_argument(
        "--interactions-file",
        help="Defines the interactions file",
        type=str,
        required=True,
    )
    plot_parser.add_argument(
        "--limit-interactions",
        help="Limits the interactions to plot",
        type=str,
        nargs="+",
    )

    plot_parser.add_argument(
        "--video-file", help="Defines the input video file", type=str, required=True
    )
    plot_parser.add_argument(
        "--models-to-plot",
        help="Defines the models to plot",
        type=str,
        nargs="+",
        required=True,
    )

    # General options
    plot_parser.add_argument(
        "--frame-start",
        help="Defines the frame to start interaction assignment",
        type=int,
        default=0,
    )
    plot_parser.add_argument(
        "--frame-end",
        help="Defines the frame to end interaction assignment",
        type=int,
        default=-1,
    )
    plot_parser.add_argument(
        "--plot-color", help="Defines the color of the models", type=str, default="red"
    )
    plot_parser.add_argument(
        "--line-width", help="Defines the line width of the models", type=int, default=2
    )

    # Output options
    plot_parser.add_argument(
        "--out-file", help="Defines the output video file", type=str, required=True
    )

    # Return the arguments
    return plot_parser.parse_args()


def main():

    # Assign the interaction args
    plot_args = plotParser()

    # Load the tracking array
    logger.info("Loading the tracking array...")
    t0 = time.time()

    # Create the tracking array object using a SLEAP file and the yaml model
    tracking_array = TrackingArray.fromSLEAP(
        plot_args.sleap_file, model_yaml=plot_args.yaml_model
    )

    logger.info("Done loading in %s seconds.", time.time() - t0)

    # Plot interactions from the tracking array
    logger.info("Starting plotting...")
    logger.info(f"Ploting frames {plot_args.frame_start} to {plot_args.frame_end}")
    t0 = time.time()

    tracking_array.plot(
        plot_args.interactions_file,
        plot_args.models_to_plot,
        plot_args.video_file,
        plot_args.out_file,
        frame_start=plot_args.frame_start,
        frame_end=plot_args.frame_end,
        limit_interactions=plot_args.limit_interactions,
    )

    logger.info("Done plotting in %s seconds.", time.time() - t0)


if __name__ == "__main__":
    main()
