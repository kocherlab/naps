from naps.sleap_utils import load_tracks_from_slp, reconstruct_slp
import argparse
import logging
import time
from naps.aruco import ArUcoModel
from naps.matching import Matching
import sleap

logger = logging.getLogger("NAPS Logger")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="NAPS -- Hybrid tracking using SLEAP and ArUco tags"
    )

    parser.add_argument(
        "--video_path",
        help="The filepath of the video used with SLEAP.",
        type=str,
        default="tests/data/example.mp4",
        # required=True,
    )

    parser.add_argument(
        "--slp_path",
        help="The filepath of the SLEAP (.slp or .h5) file to generate coordinates from, corresponding with the input video file.",
        type=str,
        default="tests/data/example.slp",
        # required=True,
    )

    parser.add_argument(
        "--tag_node",
        help="A string to include as stems of filenames saved to files_folder_path.",
        type=int,
        required=True,
    )

    args = parser.parse_args()
    return args


def main(argv=None):
    args = parse_args(argv)

    logger.info("Loading predictions...")
    t0 = time.time()
    locations = load_tracks_from_slp(args.slp_path)
    logger.info(f"Done loading predictions in {time.time() - t0} seconds.")
    tag_locations = locations[:, args.tag_node, :, :]

    logger.info("Starting matching...")
    t0 = time.time()
    matching = Matching(
        args.video_path,
        21,
        99,
        aruco_model=ArUcoModel.withTagSet("DICT_4X4_100"),
        tag_node_matrix=tag_locations,
    )
    matching_dict = matching.match()
    logger.info(f"Done matching in {time.time() - t0} seconds.")

    logger.info("Reconstructing SLEAP file...")
    t0 = time.time()
    # Right now the reconstruction assumes that we each track has a single track ID assigned to it. We'll generalize so that a track can switch IDs over time.
    resulting_labeled_frames = reconstruct_slp(args.slp_path, matching_dict)
    sleap.Labels.save_file(resulting_labeled_frames, "resulting_labeled_frames.slp")
    logger.info(f"Done reconstructing SLEAP file in {time.time() - t0} seconds.")

if __name__ == "__main__":
    main()
