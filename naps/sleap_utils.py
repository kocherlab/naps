import h5py
import numpy as np
import sleap
import pathlib
import h5py
import logging
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_location_matrix(
    labels: sleap.Labels, all_frames: bool, video: sleap.Video = None
) -> np.ndarray:
    """Builds numpy matrix with point location data.
    Note: This function assumes either all instances have tracks or no instances have
    tracks.
    Args:
        labels: The :py:class:`Labels` from which to get data.
        all_frames: If True, then includes zeros so that frame index
            will line up with columns in the output. Otherwise,
            there will only be columns for the frames between the
            first and last frames with labeling data.
        video: The :py:class:`Video` from which to get data. If no `video` is specified,
            then the first video in `source_object` videos list will be used. If there
            are no labeled frames in the `video`, then None will be returned.
    Returns:
        np.ndarray: point location array with shape (frames, nodes, 2, tracks)
    """
    # Assumes either all instances have tracks or no instances have tracks
    track_count = len(labels.tracks) or 1
    node_count = len(labels.skeletons[0].nodes)

    # Retrieve frames from current video only
    try:
        if video is None:
            video = labels.videos[0]
    except IndexError:
        print(f"There are no videos in this project. No occupancy matrix to return.")
        return
    labeled_frames = labels.get(video)

    frame_idxs = [lf.frame_idx for lf in labeled_frames]
    frame_idxs.sort()

    try:
        first_frame_idx = 0 if all_frames else frame_idxs[0]

        frame_count = (
            frame_idxs[-1] - first_frame_idx + 1
        )  # count should include unlabeled frames
    except IndexError:
        print(f"No labeled frames in {video.filename}. No occupancy matrix to return.")
        return

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

    return locations_matrix


def load_tracks_from_slp(slp_path):
    if pathlib.Path(slp_path).suffix == ".slp":
        dset = sleap.load_file(slp_path)
        locations = get_location_matrix(dset, all_frames=True)
    elif pathlib.Path(slp_path).suffix == ".h5":
        with h5py.File(slp_path, "r") as f:
            locations = f["tracks"][:].T
    return locations


def reconstruct_slp(slp_path, matching_array):
    logger.info("Loading predictions...")
    t0 = time.time()
    labels = sleap.load_file(slp_path)
    frames = sorted(labels.labeled_frames, key=operator.attrgetter("frame_idx"))
    logger.info(f"Done loading predictions in {time.time() - t0} seconds.")

    new_tracks = [
        sleap.Track(spawned_on=0, name=f"track_{i}")
        for i in range(np.unique(matching_array).shape[0])
    ]
    new_lfs = []
    for lf in tqdm(frames):
        for inst in lf.instances:
            inst.track = new_tracks[matching_array[int(inst.track.name.split("_")[-1])]]

        new_lf = sleap.LabeledFrame(
            frame_idx=lf.frame_idx, video=lf.video, instances=lf.instances
        )
        new_lfs.append(new_lf)
    return new_lfs