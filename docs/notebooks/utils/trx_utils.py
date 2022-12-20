import logging

import h5py
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import scipy.ndimage
from scipy.signal import savgol_filter
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("trx_utils_logger")


def fill_gaps(x, max_len):
    """Dilate -> erode"""
    return scipy.ndimage.binary_closing(x, structure=np.ones((max_len,)))


def connected_components1d(x, return_limits=False):
    """Return the indices of the connected components in a 1D logical array.

    Args:
        x: 1d logical (boolean) array.
        return_limits: If True, return indices of the limits of each component rather
            than every index. Defaults to False.

    Returns:
        If return_limits is False, a list of (variable size) arrays are returned, where
        each array contains the indices of each connected component.

        If return_limits is True, a single array of size (n, 2) is returned where the
        columns contain the indices of the starts and ends of each component.
    """
    L, n = scipy.ndimage.label(x.squeeze())
    ccs = scipy.ndimage.find_objects(L)
    starts = [cc[0].start for cc in ccs]
    ends = [cc[0].stop for cc in ccs]
    if return_limits:
        return np.stack([starts, ends], axis=1)
    else:
        return [np.arange(i0, i1, dtype=int) for i0, i1 in zip(starts, ends)]


def flatten_features(x, axis=0):

    if axis != 0:
        # Move time axis to the first dim
        x = np.moveaxis(x, axis, 0)

    # Flatten to 2D.
    initial_shape = x.shape
    x = x.reshape(len(x), -1)

    return x, initial_shape


def unflatten_features(x, initial_shape, axis=0):
    # Reshape.
    x = x.reshape(initial_shape)

    if axis != 0:
        # Move time axis back
        x = np.moveaxis(x, 0, axis)

    return x


def fill_missing(x, kind="nearest", axis=0, **kwargs):
    """Fill missing values in a timeseries.

    Args:
        x: Timeseries of shape (time, ...) or with time axis specified by axis.
        kind: Type of interpolation to use. Defaults to "nearest".
        axis: Time axis (default: 0).

    Returns:
        Timeseries of the same shape as the input with NaNs filled in.

    Notes:
        This uses pandas.DataFrame.interpolate and accepts the same kwargs.
    """
    if x.ndim > 2:
        # Reshape to (time, D)
        x, initial_shape = flatten_features(x, axis=axis)

        # Interpolate.
        x = fill_missing(x, kind=kind, axis=0, **kwargs)

        # Restore to original shape
        x = unflatten_features(x, initial_shape, axis=axis)

        return x

    return pd.DataFrame(x).interpolate(method=kind, axis=axis, **kwargs).to_numpy()


def smooth_ewma(x, alpha, axis=0, inplace=False):
    if axis != 0 or x.ndim > 1:
        if not inplace:
            x = x.copy()

        # Reshape to (time, D)
        x, initial_shape = flatten_features(x, axis=axis)

        # Apply function to each slice
        for i in range(x.shape[1]):
            x[:, i] = smooth_ewma(x[:, i], alpha, axis=0)

        # Restore to original shape
        x = unflatten_features(x, initial_shape, axis=axis)
        return x
    initial_shape = x.shape
    x = pd.DataFrame(x).ewm(alpha=alpha, axis=axis).mean().to_numpy()
    return x.reshape(initial_shape)


def smooth_gaussian(x, std=1, window=5, axis=0, inplace=False):
    if axis != 0 or x.ndim > 1:
        if not inplace:
            x = x.copy()

        # Reshape to (time, D)
        x, initial_shape = flatten_features(x, axis=axis)

        # Apply function to each slice
        for i in range(x.shape[1]):
            x[:, i] = smooth_gaussian(x[:, i], std, window, axis=0)

        # Restore to original shape
        x = unflatten_features(x, initial_shape, axis=axis)
        return x

    y = (
        pd.DataFrame(x.copy())
        .rolling(window, win_type="gaussian", center=True)
        .mean(std=std)
        .to_numpy()
    )
    y = y.reshape(x.shape)
    mask = np.isnan(y) & (~np.isnan(x))
    y[mask] = x[mask]
    return y


def smooth_median(x, window=5, axis=0, inplace=False):
    if axis != 0 or x.ndim > 1:
        if not inplace:
            x = x.copy()

        # Reshape to (time, D)
        x, initial_shape = flatten_features(x, axis=axis)

        # Apply function to each slice
        for i in range(x.shape[1]):
            x[:, i] = smooth_median(x[:, i], window, axis=0)

        # Restore to original shape
        x = unflatten_features(x, initial_shape, axis=axis)
        return x

    y = scipy.signal.medfilt(x.copy(), window)
    y = y.reshape(x.shape)
    mask = np.isnan(y) & (~np.isnan(x))
    y[mask] = x[mask]
    return y


def update_labels(labels, trx, copy=False):
    """Update a sleap.Labels from a tracks array.

    Args:
        labels: sleap.Labels
        trx: numpy array of shape (frames, tracks, nodes, 2)
        copy: If True, make a copy of the labels rather than updating in place.

    Return:
        The labels with updated points.
    """
    import sleap

    if copy:
        labels = labels.copy()

    for i in range(len(trx)):
        lf = labels[i]
        for j in range(trx.shape[1]):
            track_j = labels.tracks[j]
            updated_j = False
            for inst in lf:
                if inst.track != track_j:
                    continue
                for node, pt in zip(labels.skeleton.nodes, trx[i, j]):
                    if np.isnan(pt).all():
                        continue
                    inst[node].x, inst[node].y = pt
                updated_j = True
            if not updated_j and not np.isnan(trx[i, j]).all():
                inst = sleap.Instance.from_numpy(
                    trx[i, j], labels.skeleton, track=track_j
                )
                inst.frame = lf
                lf.instances.append(inst)

    return labels


def normalize_to_egocentric(
    x, rel_to=None, scale_factor=1, ctr_ind=1, fwd_ind=0, fill=True, return_angles=False
):
    """Normalize pose estimates to egocentric coordinates.

    Args:
        x: Pose of shape (joints, 2) or (time, joints, 2)
        rel_to: Pose to align x with of shape (joints, 2) or (time, joints, 2). Defaults
            to x if not specified.
        scale_factor: Spatial scaling to apply to coordinates after centering.
        ctr_ind: Index of centroid joint. Defaults to 1.
        fwd_ind: Index of "forward" joint (e.g., head). Defaults to 0.
        fill: If True, interpolate missing ctr and fwd coordinates. If False, timesteps
            with missing coordinates will be all NaN. Defaults to True.
        return_angles: If True, return angles with the aligned coordinates.

    Returns:
        Egocentrically aligned poses of the same shape as the input.

        If return_angles is True, also returns a vector of angles.
    """

    if rel_to is None:
        rel_to = x

    is_singleton = (x.ndim == 2) and (rel_to.ndim == 2)

    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    if rel_to.ndim == 2:
        rel_to = np.expand_dims(rel_to, axis=0)

    # Find egocentric forward coordinates.
    ctr = rel_to[..., ctr_ind, :]  # (t, 2)
    fwd = rel_to[..., fwd_ind, :]  # (t, 2)
    if fill:
        ctr = fill_missing(ctr, kind="nearest")
        fwd = fill_missing(fwd, kind="nearest")
    ego_fwd = fwd - ctr

    # Compute angle.
    ang = np.arctan2(
        ego_fwd[..., 1], ego_fwd[..., 0]
    )  # arctan2(y, x) -> radians in [-pi, pi]
    ca = np.cos(ang)  # (t,)
    sa = np.sin(ang)  # (t,)

    # Build rotation matrix.
    rot = np.zeros([len(ca), 3, 3], dtype=ca.dtype)
    rot[..., 0, 0] = ca
    rot[..., 0, 1] = -sa
    rot[..., 1, 0] = sa
    rot[..., 1, 1] = ca
    rot[..., 2, 2] = 1

    # Center and scale.
    x = x - np.expand_dims(ctr, axis=1)
    x /= scale_factor

    # Pad, rotate and crop.
    x = np.pad(x, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1) @ rot
    x = x[..., :2]

    if is_singleton:
        x = x[0]

    if return_angles:
        return x, ang
    else:
        return x


def instance_node_velocities(fly_node_locations, start_frame, end_frame):
    frame_count = len(range(start_frame, end_frame))
    if len(fly_node_locations.shape) == 4:
        fly_node_velocities = np.zeros(
            (frame_count, fly_node_locations.shape[1], fly_node_locations.shape[3])
        )
        for fly_idx in range(fly_node_locations.shape[3]):
            for n in tqdm(range(0, fly_node_locations.shape[1])):
                fly_node_velocities[:, n, fly_idx] = smooth_diff(
                    fly_node_locations[start_frame:end_frame, n, :, fly_idx]
                )
    else:
        fly_node_velocities = np.zeros((frame_count, fly_node_locations.shape[1]))
        for n in tqdm(range(0, fly_node_locations.shape[1] - 1)):
            fly_node_velocities[:, n] = smooth_diff(
                fly_node_locations[start_frame:end_frame, n, :]
            )

    return fly_node_velocities


def smooth_diff(node_loc, win=25, poly=3):
    """
    node_loc is a [frames, 2] arrayF

    win defines the window to smooth over

    poly defines the order of the polynomial
    to fit with

    """
    node_loc_vel = np.zeros_like(node_loc)

    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = np.gradient(
            node_loc[:, c]
        )  # savgol_filter(node_loc[:, c], win, poly, deriv=1)

    node_vel = np.linalg.norm(node_loc_vel, axis=1)

    return node_vel


from scipy.interpolate import interp1d
from tqdm import tqdm


def fill_missing_np(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))
    # Interpolate along each slice.
    for i in tqdm(range(Y.shape[-1])):
        try:
            y = Y[:, i]

            # Build interpolant.
            x = np.flatnonzero(~np.isnan(y))
            f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

            # Fill missing
            xq = np.flatnonzero(np.isnan(y))
            y[xq] = f(xq)

            # Fill leading or trailing NaNs with the nearest non-NaN values
            mask = np.isnan(y)
            y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

            # Save slice
            Y[:, i] = y
        except:
            Y[:, i] = 0
    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y


def hist_sort(
    locations, ctr_idx, xbins=2, ybins=2, xmin=0, xmax=1536, ymin=0, ymax=1536
):
    assignments = []
    freq = []
    for track_num in range(locations.shape[3]):
        h, xedges, yedges = np.histogram2d(
            locations[:, ctr_idx, 0, track_num],
            locations[:, ctr_idx, 1, track_num],
            range=[[xmin, xmax], [ymin, ymax]],
            bins=[xbins, ybins],
        )
        # Probably not the correct way to flip this around to get rows and columns correct but it'll do!
        assignment = h.argmax()
        freq.append(h)
        assignments.append(assignment)
    assignment_indices = np.argsort(assignments)
    locations = locations[:, :, :, assignment_indices]
    return assignment_indices, locations, freq


import scipy.stats


def hist_sort_rowwise(
    locations, ctr_idx, xbins=2, ybins=2, xmin=0, xmax=1536, ymin=0, ymax=1536
):
    fourbyfour_interior_gridmap = {5: 0, 6: 1, 9: 2, 10: 3}
    output = np.zeros_like(locations)
    for track_num in range(locations.shape[3]):
        h, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(
            locations[:, ctr_idx, 0, track_num],
            locations[:, ctr_idx, 1, track_num],
            None,
            "count",
            range=[[xmin, xmax], [ymin, ymax]],
            bins=[xbins, ybins],
        )
        assignments = np.vectorize(fourbyfour_interior_gridmap.get)(binnumber)
        frames, nodes, coords, assignments = (
            np.arange(locations.shape[0]),
            slice(None),
            slice(None),
            assignments,
        )
        output[frames, nodes, coords, assignments] = locations[
            frames, nodes, coords, assignments
        ]
    return output


def acorr(x, normed=True, maxlags=10):
    return xcorr(x, x, normed, maxlags)


def xcorr(x, y, normed=True, maxlags=10):

    Nx = len(x)
    if Nx != len(y):
        raise ValueError("x and y must be equal length")

    correls = np.correlate(x, y, mode="full")

    if normed:
        correls /= np.sqrt(np.dot(x, x) * np.dot(y, y))

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError("maxlags must be None or strictly " "positive < %d" % Nx)
    lags = np.arange(-maxlags, maxlags + 1)
    correls = correls[Nx - 1 - maxlags : Nx + maxlags]
    return lags, correls


def isintimeperiod(startTime, endTime, nowTime):
    if startTime < endTime:
        return nowTime >= startTime and nowTime <= endTime
    else:
        return nowTime >= startTime or nowTime <= endTime


def alpha_cmap(cmap):
    my_cmap = cmap(np.arange(cmap.N))
    # Set a square root alpha.
    x = np.linspace(0, 1, cmap.N)
    my_cmap[:, -1] = x ** (0.5)
    my_cmap = colors.ListedColormap(my_cmap)
    return my_cmap


import cv2


def blob_detector(video_path):

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for f in np.random.choice(np.arange(frame_count), 20, replace=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    th, im_th = cv2.threshold(median_frame, 100, 255, cv2.THRESH_BINARY)

    # # Copy the thresholded image.
    # im_floodfill = im_th.copy()
    # h, w = im_th.shape[:2]
    # mask = np.zeros((h+2, w+2), np.uint8)
    # cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # im_out = im_th | im_floodfill_inv

    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    # params.minThreshold = 10
    # params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = False
    params.minArea = 10

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(255 - im_th)

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(
        im_th, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return keypoints, blobs, median_frame


import copyreg


def _pickle_keypoints(point):
    return cv2.KeyPoint, (
        *point.pt,
        point.size,
        point.angle,
        point.response,
        point.octave,
        point.class_id,
    )


copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)
import matplotlib.pyplot as plt


def drawArrow(A, B):
    plt.arrow(
        A[0], A[1], B[0] - A[0], B[1] - A[1], head_width=3, length_includes_head=True
    )


def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * 0.5)


def describe_hdf5(filename, attrs=True):
    def desc(k, v):
        if type(v) == h5py.Dataset:
            print(f"[dataset]  {v.name}: {v.shape} | dtype = {v.dtype}")
            if attrs and len(v.attrs) > 0:
                print(f"      attrs = {dict(v.attrs.items())}")
        elif type(v) == h5py.Group:
            print(f"[group] {v.name}:")
            if attrs and len(v.attrs) > 0:
                print(f"      attrs = {dict(v.attrs.items())}")

    with h5py.File(filename, "r") as f:
        f.visititems(desc)


import palettable
import skvideo

skvideo.setFFmpegPath("/Genomics/argo/users/swwolf/.conda/envs/sleap_dev/bin")
import skvideo.io


def plot_trx(
    tracks,
    video_path,
    frame_start=0,
    frame_end=100,
    trail_length=10,
    output_path="output.mp4",
):
    # ffmpeg_writer = skvideo.io.FFmpegWriter(
    #     f"{output_path}_fly_node_locations.mp4",
    #     outputdict={
    #         "-vcodec": "libx265",
    #         "-r": "20",
    #         "-pix_fmt": "yuv420p",
    #         "-b:v": "50M",
    #         "-crf": "1",
    #         "-preset": "veryslow",
    #     },
    # )
    # cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start - 1)
    data = tracks[frame_start:frame_end, :, :, :]
    dpi = 1000
    pal = sns.husl_palette(14)
    for frame_idx in range(data.shape[0]):
        fig, ax = plt.subplots(figsize=(1536 / 300, 1536 / 300), dpi=dpi)
        print(f"Frame {frame_idx}")
        data_subset = data[max((frame_idx - trail_length), 0) : (frame_idx), :, :, :]
        # if frame_idx >= 150:
        #     if trail_length <= 5:
        #         trail_length = 5
        #         continue
        #     else:
        #         trail_length = 150 - ((frame_idx - 150)*3)
        for fly_idx in range(data_subset.shape[3]):
            for node_idx in range(data_subset.shape[1]):
                for idx in range(2, data_subset.shape[0]):
                    if trail_length != 1:
                        # Note that you need to use single steps or the data has "steps"
                        plt.plot(
                            data_subset[(idx - 1) : (idx + 1), node_idx, 0, fly_idx],
                            data_subset[(idx - 1) : (idx + 1), node_idx, 1, fly_idx],
                            linewidth=0.5 * idx / data_subset.shape[0],
                            color=pal[node_idx],
                            alpha=0.3,
                            solid_capstyle="round",
                        )
                    else:
                        plt.scatter(
                            data_subset[idx, node_idx, 0, fly_idx],
                            data_subset[idx, node_idx, 1, fly_idx],
                            color=palettable.tableau.Tableau_20.mpl_colors[node_idx],
                        )

        # if cap.isOpened():
        #     res, frame = cap.read()
        #     frame = frame[:, :, 0]
        #     plt.imshow(frame, cmap="gray")

        # plt.xlim(0,1535)
        # plt.ylim(0,1535)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        fig.set_size_inches(5, 5, True)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis("off")
        fig.patch.set_visible(False)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )
        # ffmpeg_writer.writeFrame(image_from_plot)
        plt.close()
    # ffmpeg_writer.close()
    # return fig


import cv2
import numpy as np


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection


def plot_ego(
    tracks,
    video_path,
    angles=None,
    fly_ids=[1],
    ctr_idx=3,
    frame_start=0,
    frame_end=100,
    trail_length=50,
    output_path="output.mp4",
):
    if isinstance(fly_ids, int):
        fly_idx = [fly_ids]
    ffmpeg_writer = skvideo.io.FFmpegWriter(
        f"{output_path}",
        outputdict={
            "-vcodec": "libx264",
            "-r": "25",
            "-pix_fmt": "yuv420p",
            "-b:v": "20M",
            "-crf": "1",
            "-preset": "slow",
        },
    )
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start - 1)
    dpi = 300
    if tracks.ndim == 3:
        tracks = tracks[:, :, :, np.newaxis]
    data = tracks[frame_start:frame_end, :, :, :]
    print(data.shape)
    for frame_idx in range(data.shape[0]):
        if cap.isOpened():
            res, frame = cap.read()
            frame = frame[:, :, 0].astype(float)

            height, width, nbands = 96 * 5, 96 * 5, 3
            # What size does the figure need to be in inches to fit the image?
            figsize = width / float(dpi), height / float(dpi)
            # Create a figure of the right size with one axes that takes up the full figure
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0, 0, 1, 1])
            # Hide spines, ticks, etc.
            ax.axis("off")
            ax.imshow(frame, cmap="gray")
        else:
            fig, ax = plt.subplots()
        print(f"Frame {frame_idx}")
        data_subset = data[max((frame_idx - trail_length), 0) : frame_idx, :, :, :]
        if data_subset.ndim == 4:
            for fly_idx in fly_ids:
                for node_idx in range(data_subset.shape[1]):
                    x = data_subset[:, node_idx, 0, fly_idx]
                    y = data_subset[:, node_idx, 1, fly_idx]
                    lwidths = np.arange(data_subset.shape[0]) * 2
                    points = np.array([x, y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    alphas = np.linspace(0.1, 1, data_subset.shape[0])
                    rgba_colors = np.zeros((data_subset.shape[0], 4))
                    # for red the first column needs to be one
                    for i in range(len(x)):
                        rgba_colors[i, 0] = palettable.tableau.Tableau_20.mpl_colors[
                            node_idx
                        ][0]
                        rgba_colors[i, 1] = palettable.tableau.Tableau_20.mpl_colors[
                            node_idx
                        ][1]
                        rgba_colors[i, 2] = palettable.tableau.Tableau_20.mpl_colors[
                            node_idx
                        ][2]
                        rgba_colors[i, 3] = alphas[i]
                    # print(rgba_colors)
                    lc = LineCollection(
                        segments,
                        linewidths=lwidths / data_subset.shape[0],
                        colors=rgba_colors,
                        path_effects=[path_effects.Stroke(capstyle="round")],
                    )  # =palettable.tableau.Tableau_20.mpl_colors[node_idx])
                    ax.add_collection(lc)

        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)
        # fig.set_size_inches(5, 5, True);
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # ax.axis('off')
        # fig.patch.set_visible(False)
        # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax.set(xlim=[-0.5, width - 0.5], ylim=[height - 0.5, -0.5], aspect=1)
        xy = data[frame_idx, ctr_idx, 0:2, fly_idx]
        x = xy[0]
        y = xy[1]
        ax.set_xlim((x - 96), (x + 96))
        ax.set_ylim((y - 96), (y + 96))

        image_from_plot = get_img_from_fig(fig, dpi)
        # image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if angles is not None:
            # print(angles)
            image_from_plot = rotate_image(
                image_from_plot, -angles[frame_idx + frame_start] * 180 / np.pi
            )
            image_from_plot = image_from_plot[64:-64, 64:-64, :]
        ffmpeg_writer.writeFrame(image_from_plot)
        plt.close()
        # fig.close()
    ffmpeg_writer.close()


import io


def get_img_from_fig(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def ortho_bounding_box(point_list):
    x_min = min(point[0] for point in point_list)
    x_max = max(point[0] for point in point_list)
    y_min = min(point[1] for point in point_list)
    y_max = max(point[1] for point in point_list)
    return x_min, x_max, y_min, y_max


def rotate(p, origin=(0, 0), angle=0):
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_timeseries(
    tracks,
    fly_idx,
    vels,
    node_idx,
    frame_start=0,
    frame_end=100,
    title="",
    output_name="",
    path="",
):
    if isinstance(node_idx, int):
        node_idx = [node_idx]
    if isinstance(fly_idx, int):
        fly_idx = [fly_idx]
    dpi = 300
    data = tracks[frame_start:frame_end, node_idx, :, :]
    sns.set("notebook", "ticks", font_scale=1.2)
    plt.rcParams["figure.figsize"] = [16, 4]
    for fly_idx in fly_idx:
        if vels is not None:
            fig, ax = plt.subplots(
                3, sharex=True
            )  # (figsize=(1536/300,1536/300),dpi=dpi)
            fig.tight_layout()
            x = data[:, :, 0, fly_idx]
            y = data[:, :, 1, fly_idx]
            ax[0].plot(
                np.arange(frame_start, frame_end),
                x,
                label="x",
                color=palettable.wesanderson.Cavalcanti_5.mpl_colors[0],
            )
            ax[1].plot(
                np.arange(frame_start, frame_end),
                y,
                label="y",
                color=palettable.wesanderson.Cavalcanti_5.mpl_colors[1],
            )
            ax[0].set_title(f"{title} fly {fly_idx} position by time")
            ax[0].margins(x=0)
            ax[1].margins(x=0)
            vel_img = vels[frame_start:frame_end, node_idx[0], fly_idx][:, np.newaxis].T
            im = ax[2].imshow(
                vel_img,
                vmin=np.min(vels),
                vmax=np.max(vel_img),
                extent=[frame_start, frame_end, 300, 0],
            )  # ,cmap='jet')
            # fig.mar
            # Be sure to only pick integer tick locations.
            for axis in ax[0:2]:
                axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                axis.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            fig.text(
                -0.02, 0.66, "Position (pixels)", ha="center", va="center", rotation=90
            )
            fig.text(0.51, 0, "Time (frames)", ha="center", va="center")
            fig.text(-0.02, 0.45 / 2, "Rel vel", ha="center", va="center", rotation=90)

            ax[2].set_title(f"{title} fly {fly_idx} velocity by time")
            divider = make_axes_locatable(ax[2])
            # cax = divider.append_axes('right', size='5%', pad=0.05)
            # fig.colorbar(im, cax=cax, orientation='vertical')
            fig.savefig(
                f"{path}{output_name}_fly_{fly_idx}_{frame_start}to{frame_end}.png",
                dpi=dpi,
            )
            plt.show()

        else:
            fig, ax = plt.subplots(
                2, sharex=True
            )  # (figsize=(1536/300,1536/300),dpi=dpi)
            fig.tight_layout()
            x = data[:, :, 0, fly_idx]
            y = data[:, :, 1, fly_idx]
            ax[0].plot(
                np.arange(frame_start, frame_end),
                x,
                label="x",
                color=palettable.wesanderson.Cavalcanti_5.mpl_colors[0],
            )
            ax[1].plot(
                np.arange(frame_start, frame_end),
                y,
                label="y",
                color=palettable.wesanderson.Cavalcanti_5.mpl_colors[1],
            )
            ax[0].set_title(f"{title} fly {fly_idx} - thorax position by time")
            ax[0].margins(x=0)
            ax[1].margins(x=0)
            ax[1].set_xlabel("Time (frames)")
            ax[0].set_ylabel("x position (px)")
            ax[1].set_ylabel("y position (px)")
            # Be sure to only pick integer tick locations.
            for axis in ax:
                axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                axis.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            fig.savefig(
                f"{path}{output_name}_fly_{fly_idx}_{frame_start}to{frame_end}.png",
                dpi=dpi,
            )
            plt.show()
        # plt.legend()


def load_tracks(expmt_dict):
    tracks_dict_raw = {}
    for key in expmt_dict:
        expmt_name = str(key)
        logger.info(f"Loading {expmt_name}")
        expmt = expmt_dict[key]

        with h5py.File(expmt["h5s"][0], "r") as f:
            logger.info(expmt["h5s"][0])
            dset_names = list(f.keys())
            # Note the assignment of node_names here!
            node_names = [n.decode() for n in f["node_names"][:]]
            expmt_dict[key]["node_names"] = node_names
            locations = f["tracks"][:].T

            locations[:, :, 1, :] = -locations[:, :, 1, :]
            assignment_indices, locations, freq = hist_sort(
                locations, ctr_idx=node_names.index("thorax"), ymin=-1536, ymax=0
            )
            locations[:, :, 1, :] = -locations[:, :, 1, :]

        if len(expmt["h5s"]) > 1:
            for filename in tqdm(expmt["h5s"][1:]):
                with h5py.File(filename, "r") as f:
                    temp_locations = f["tracks"][:].T
                    temp_locations[:, :, 1, :] = -temp_locations[:, :, 1, :]
                    temp_assignment_indices, temp_locations, freq = hist_sort(
                        temp_locations,
                        ctr_idx=node_names.index("thorax"),
                        ymin=-1536,
                        ymax=0,
                    )
                    temp_locations[:, :, 1, :] = -temp_locations[:, :, 1, :]

                    # logger.info(filename)
                    # logger.info(freq)

                    locations = np.concatenate((locations, temp_locations), axis=0)

        # Final assignment
        locations[:, :, 1, :] = -locations[:, :, 1, :]
        assignment_indices, locations, freq = hist_sort(
            locations, ctr_idx=node_names.index("thorax"), ymin=-1536, ymax=0
        )
        locations[:, :, 1, :] = -locations[:, :, 1, :]
        logger.info(f"Experiment: {str(expmt)}")
        logger.info(f"Final frequencies: {freq}")
        logger.info(f"Final assignments: {assignment_indices}")

        tracks_dict_raw[expmt_name] = locations  # [0:1000,:,:,:]
        expmt_dict[expmt_name]["assignments"] = assignment_indices
        expmt_dict[expmt_name]["freq"] = freq
    return expmt_dict, tracks_dict_raw


from pathlib import Path


def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def rle(inarray):
    """run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, startpositions, values)"""
    ia = np.asarray(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return (z, p, ia[i])


def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window - 1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad, mode="reflect")
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
