#!/usr/bin/env python
import argparse
import logging
import time
from collections import defaultdict

import sleap

from naps.aruco import ArUcoModel
from naps.matching import Matching
from naps.sleap_utils import update_labeled_frames

logger = logging.getLogger("NAPS Logger")


def build_parser():
    parser = argparse.ArgumentParser(
        description="NAPS -- Hybrid tracking using SLEAP and ArUco tags"
    )

    parser.add_argument(
        "--slp-path",
        help="The filepath of the SLEAP (.slp or .h5) file to pull coordinates from. This should correspond with the input video file. The SLEAP file contains pose estimation data required for NAPS.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--h5-path",
        help="The filepath of the analysis h5 file to pull coordinates from. This should correspond with the input video file and slp file. Because NAPS relies on having data in the format given by SLEAP analysis h5s, you can skip a step by providing the analysis h5 file directly. If this is not provided, NAPS will pull the data from the SLEAP file.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--video-path",
        help="The filepath of the video used with SLEAP. This should be the video file that SLEAP was run on.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--tag-node-name",
        help="The ArUco tag SLEAP node name. This should correspond to the node name of the ArUco tag in the SLEAP data. 'tag' is set as the default value.",
        type=str,
        default="tag",
    )

    parser.add_argument(
        "--start-frame",
        help="The zero-based fully-closed frame to begin NAPS assignment. This value allows you to specify from which frame to start the processing.",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--end-frame",
        help="The zero-based fully-closed frame to stop NAPS assignment. This allows you to specify at which frame to end the processing.",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--aruco-marker-set",
        help="The ArUco markers used in the video. This must match the specific set of ArUco markers used in the video for accurate detection and tracking. An example would be DICT_5X5_50, which is a set of 50 5x5 markers. More information can be found in the OpenCV documentation.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--half-rolling-window-size",
        help="Specifies the number of flanking frames (prior and subsequent) required in the rolling window for Hungarian matching a frame. The larger this window, the more frames will be used for matching. This can result in more robust tracks but in the event of an identity swap, it may take longer to correct the identity. The default is set to 5, resulting in an 11 frame window. You should change this depending on your intended use and your recording setup. ",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output-path",
        help="Output path of the resulting SLEAP h5 analysis file. The default value is 'naps_output.analysis.h5', storing the output in the current working directory. If a different location or file name is preferred, it can be specified here.",
        type=str,
        default="naps_output.analysis.h5",
    )

    parser.add_argument(
        "--aruco-crop-size",
        help="The number of pixels horizontally and vertically around the ArUco SLEAP node to identify the marker. The cropping area should be large enough to include the whole marker for accurate detection, but not much larger, to keep the processing efficient and avoid capturing multiple tags. This is data set specific.",
        type=int,
    )

    parser.add_argument(
        "--aruco-adaptive-thresh-win-size-min",
        dest="adaptiveThreshWinSizeMin",
        help="Specifies the value for adaptiveThreshWinSizeMin used in adaptive thresholding. This parameter affects the adaptive thresholding in the ArUco marker detection, which can impact the robustness of marker detection. The default value is 10, a commonly used value. More information can be found in the OpenCV documentation.",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--aruco-adaptive-thresh-win-size-max",
        dest="adaptiveThreshWinSizeMax",
        help="Specifies the value for adaptiveThreshWinSizeMax used in adaptive thresholding. This parameter, similar to adaptiveThreshWinSizeMin, influences the adaptive thresholding. The default value is 30, providing a larger window size for the thresholding process, improving marker detection under diverse lighting conditions. More information can be found in the OpenCV documentation.",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--aruco-adaptive-thresh-win-size-step",
        dest="adaptiveThreshWinSizeStep",
        help="Specifies the value for adaptiveThreshWinSizeStep used in adaptive thresholding. This parameter determines the step size for the window in adaptive thresholding, affecting the granularity of the process. The default value is 12, offering a balanced choice between processing speed and thresholding precision. More information can be found in the OpenCV documentation.",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--aruco-adaptive-thresh-constant",
        dest="adaptiveThreshConstant",
        help="Specifies the value for adaptiveThreshConstant used in adaptive thresholding. This parameter is a constant subtracted from the mean or weighted sum of the neighbourhood pixels. The default value is 3, which works well in most scenarios, but can be adjusted based on specific lighting conditions. More information can be found in the OpenCV documentation.",
        type=float,
        default=3,
    )

    parser.add_argument(
        "--aruco-perspective-rm-ignored-margin",
        dest="perspectiveRemoveIgnoredMarginPerCell",
        help="Specifies the value for perspectiveRemoveIgnoredMarginPerCell. This parameter is used in the perspective removal of the marker. The default value is 0.1, which is a reasonable value for many situations. More information can be found in the OpenCV documentation.",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--aruco-error-correction-rate",
        dest="errorCorrectionRate",
        help="Specifies the value for errorCorrectionRate. This parameter is used for error correction when decoding ArUco tags. The default value is 1 is to fully utilize the error capability for each tag dictionary. Adjust this parameter can be necessary for improving the robustness of marker detection, especially in the presence of camera noise or occlusions. More information can be found in the OpenCV documentation.",
        type=float,
        default=1,
    )

    return parser


def main(argv=None):
    """Main function for the NAPS tracking script."""

    # Set the start time for the entire pipeline
    t0_total = time.time()

    # Build the arguments from the .parse_args(args)
    args = build_parser().parse_args(argv)

    # Assign the h5 path if not specified
    if args.h5_path is None:
        args.h5_path = args.slp_path

    # Create a track array from the SLEAP file(s)
    logger.info("Loading predictions...")
    t0 = time.time()
    # locations, node_names = load_tracks_from_slp(args.h5_path)
    tag_locations_dict = defaultdict(lambda: defaultdict(tuple))
    labels = sleap.Labels.load_file(args.slp_path)
    for lf in labels.labeled_frames:
        if lf.frame_idx < args.start_frame or lf.frame_idx > args.end_frame:
            continue
        for instance in lf.instances:
            tag_idx = instance.skeleton.node_names.index(args.tag_node_name)
            track_name = int(instance.track.name.split("_")[-1])
            # print(f'Frame {lf.frame_idx} track {track_name} has points {instance.numpy()[tag_idx]}')
            tag_locations_dict[lf.frame_idx][track_name] = instance.numpy()[tag_idx]

    # Create an ArUcoModel with the default/specified parameters
    logger.info("Create ArUco model...")
    t0 = time.time()
    aruco_model = ArUcoModel.withTagSet(
        args.aruco_marker_set,
        adaptiveThreshWinSizeMin=args.adaptiveThreshWinSizeMin,
        adaptiveThreshWinSizeMax=args.adaptiveThreshWinSizeMax,
        adaptiveThreshWinSizeStep=args.adaptiveThreshWinSizeStep,
        adaptiveThreshConstant=args.adaptiveThreshConstant,
        perspectiveRemoveIgnoredMarginPerCell=args.perspectiveRemoveIgnoredMarginPerCell,
        errorCorrectionRate=args.errorCorrectionRate,
    )
    logger.info("ArUco model built in %s seconds.", time.time() - t0)

    # Match the track to the ArUco markers
    logger.info("Starting matching...")
    t0 = time.time()
    matching = Matching(
        args.video_path,
        args.start_frame,
        args.end_frame,
        marker_detector=aruco_model.detect,
        aruco_crop_size=args.aruco_crop_size,
        half_rolling_window_size=args.half_rolling_window_size,
        tag_node_dict=tag_locations_dict,
    )
    matching_dict = matching.match()
    logger.info("Done matching in %s seconds.", time.time() - t0)

    # Create the output
    logger.info("Reconstructing SLEAP file...")
    t0 = time.time()
    # Right now the reconstruction assumes that we each track has a single track ID assigned to it. We'll generalize so that a track can switch IDs over time.
    labels = update_labeled_frames(
        args.slp_path, matching_dict, args.start_frame, args.end_frame
    )
    labels.save(args.output_path)
    logger.info("Done reconstructing SLEAP file in %s seconds.", time.time() - t0)

    logger.info("Complete NAPS runtime: %s", time.time() - t0_total)


if __name__ == "__main__":
    main()
