"""
    -------------------------------------------------------------------
    sleap_utils.py: General helper functions for interfacing with SLEAP
    -------------------------------------------------------------------
"""

import itertools
import logging
import pathlib
from typing import List, Tuple

import attr
import h5py
import numpy as np
import sleap

logger = logging.getLogger(__name__)


# Pulled from SLEAP's codebase
def get_location_matrix(
    labels: sleap.Labels, all_frames: bool, video: sleap.Video = None
) -> Tuple[np.ndarray, List]:
    """Builds numpy matrix with point location data. Note: This function assumes either all instances have tracks or no instances have tracks.

    Args:
        labels (sleap.Labels): The :py:class:`sleap.Labels` from which to get data.
        all_frames (bool): If True, then includes zeros so that frame index will line up with columns in the output. Otherwise, there will only be columns for the frames between the first and last frames with labeling data.
        video (sleap.Video, optional): The :py:class:`Video` from which to get data. If no `video` is specified, then the first video in `source_object` videos list will be used. If there are no labeled frames in the `video`, then None will be returned. Defaults to None.

    Returns:
        Tuple[np.ndarray, List[str]]: point location array with shape (frames, nodes, 2, tracks) List[str]: list of node names
    """
    # Assumes either all instances have tracks or no instances have tracks
    track_count = len(labels.tracks) or 1
    node_count = len(labels.skeletons[0].nodes)

    # Retrieve frames from current video only
    try:
        if video is None:
            video = labels.videos[0]
    except IndexError as e:
        print("There are no videos in this project. No occupancy matrix to return.")
        raise e
    labeled_frames = labels.get(video)

    frame_idxs = [lf.frame_idx for lf in labeled_frames]
    frame_idxs.sort()

    try:
        first_frame_idx = 0 if all_frames else frame_idxs[0]

        frame_count = (
            frame_idxs[-1] - first_frame_idx + 1
        )  # count should include unlabeled frames
    except IndexError as e:
        print(f"No labeled frames in {video.filename}. No occupancy matrix to return.")
        raise e

    # Desired MATLAB format:
    # "tracks"              frames * nodes * 2 * tracks
    # "track_names"         tracks

    locations_matrix = np.full(
        (frame_count, node_count, 2, track_count), np.nan, dtype=float
    )

    # Assumes either all instances have tracks or no instances have tracks
    # Prefer user-labeled instances over predicted instances
    tracks = labels.tracks or [None]  # Comparator in case of project with no tracks
    lfs_instances = list()
    warning_flag = False
    for lf in labeled_frames:

        user_instances = lf.user_instances
        predicted_instances = lf.predicted_instances
        for track in tracks:
            track_instances = list()
            # If a user-instance exists for this track, then use user-instance
            user_track_instances = [
                inst for inst in user_instances if inst.track == track
            ]
            if len(user_track_instances) > 0:
                track_instances = user_track_instances
            else:
                # Otherwise, if a predicted instance exists, then use the predicted
                predicted_track_instances = [
                    inst for inst in predicted_instances if inst.track == track
                ]
                if len(predicted_track_instances) > 0:
                    track_instances = predicted_track_instances

            lfs_instances.extend([(lf, inst) for inst in track_instances])

            # Set warning flag if more than one instances on a track in a single frame
            warning_flag = warning_flag or (
                (track is not None) and (len(track_instances) > 1)
            )

    if warning_flag:
        print(
            "\nWarning! "
            "There are more than one instances per track on a single frame.\n"
        )

    for lf, inst in lfs_instances:
        frame_i = lf.frame_idx - first_frame_idx
        # Assumes either all instances have tracks or no instances have tracks
        if inst.track is None:
            # We could use lf.instances.index(inst) but then we'd need
            # to calculate the number of "tracks" based on the max number of
            # instances in any frame, so for now we'll assume that there's
            # a single instance if we aren't using tracks.
            track_i = 0
        else:
            track_i = labels.tracks.index(inst.track)

        locations_matrix[frame_i, ..., track_i] = inst.numpy()
    return locations_matrix, labels.skeletons[0].nodes


def load_tracks_from_slp(slp_path: str) -> Tuple[np.ndarray, List[str]]:
    """Loads tracks from a sleap project file or analysis h5

    Args:
        slp_path (str): Path to SLEAP project file or analysis h5. This file is typically generated using `sleap-convert`. Analysis h5s are much faster to load and thus preferable.

    Returns:
        Tuple[np.ndarray, List[str]]: Tuple of (locations_matrix, node_names) where location matrix is a numpy array with shape (frames, nodes, 2, tracks) and node_names is a list of node names.
    """
    if pathlib.Path(slp_path).suffix == ".slp":
        dset = sleap.load_file(slp_path)
        locations, node_names = get_location_matrix(dset, all_frames=True)
        node_names = [node.name for node in node_names]
    elif pathlib.Path(slp_path).suffix == ".h5":
        with h5py.File(slp_path, "r") as f:
            locations = f["tracks"][:].T
            node_names = [n.decode() for n in f["node_names"][:]]
    return locations, node_names


def update_labeled_frames(
    slp_path: str, matching_dict: dict, first_frame_idx: int, last_frame_idx: int
) -> List[sleap.LabeledFrame]:
    """Generates a list of labeled frames from a sleap project file with tracks updated from the provided matching_dict.

    Args:
        slp_path (str): Path to SLEAP project file (.slp) file.
        matching_dict (dict): Dictionary of format [Frame][Track] containing the tag number for each track in each frame.
        first_frame_idx (int): First frame index to include in the reconstructed h5.
        last_frame_idx (int): Last frame index to include in the reconstructed h5.

    Returns:
        List[sleap.LabeledFrame]: List of labeled frames with updated tracks.
    """
    labels = sleap.load_file(slp_path)
    # frames = sorted(labels.labeled_frames, key=operator.attrgetter("frame_idx"))

    # Matching dict is of the form [Frame][Track][Tag]
    # List of lists of tracks
    tags = [track_dict.values() for track_dict in matching_dict.values()]

    # Set of all tags
    tags = set(itertools.chain(*tags))
    logger.info("Total tags: %s", len(tags))
    new_tracks = {
        tag: sleap.Track(spawned_on=0, name=f"ArUcoTag#{tag}") for tag in tags
    }
    new_lfs = []

    # Run tracking on every frame
    for lf in labels.labeled_frames[first_frame_idx : last_frame_idx + 1]:
        tracked_instances = []
        for inst in lf.instances:
            try:
                inst_name = int(inst.track.name.split("_")[-1])
                track = matching_dict[lf.frame_idx][inst_name]
                tracked_instances.append(attr.evolve(inst, track=new_tracks[track]))
            except Exception as e:
                tracked_instances.append(attr.evolve(inst, track=None))
                continue
        new_lfs.append(attr.evolve(lf, instances=tracked_instances))
    tracked_labels = sleap.Labels(new_lfs)
    return tracked_labels
