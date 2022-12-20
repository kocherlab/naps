# Load predictions
import sleap

sleap.versions()
sleap.system_summary()

PREDICTIONS_FILE =
labels = sleap.load_file(PREDICTIONS_FILE)

# Here I'm removing the tracks so we just have instances without any tracking applied.
for instance in labels.instances():
    instance.track = None
labels.tracks = []

# Create tracker
tracker = sleap.nn.tracking.Tracker.make_tracker_by_name(
    # General tracking options
    tracker="flow",
    track_window=3,

    # Matching options
    similarity="instance",
    match="greedy",
    min_new_track_points=1,
    min_match_points=1,

    # Optical flow options (only applies to "flow" tracker)
    img_scale=0.5,
    of_window_size=21,
    of_max_levels=3,

    # Pre-tracking filtering options
    target_instance_count=2,
    pre_cull_to_target=True,
    pre_cull_iou_threshold=0.8,

    # Post-tracking filtering options
    post_connect_single_breaks=True,
    clean_instance_count=0,
    clean_iou_threshold=None,
)

tracked_lfs = []
for lf in labels:
    lf.instances = tracker.track(lf.instances, img=lf.image)
    tracked_lfs.append(lf)
tracked_labels = sleap.Labels(tracked_lfs)

tracked_labels.save("retracked.slp")
