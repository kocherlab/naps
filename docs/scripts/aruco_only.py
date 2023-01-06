# %%
import sys

import cv2
import matplotlib as mpl
import numpy as np

sys.path.append("..")

mpl.rcParams["figure.facecolor"] = "w"
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.dpi"] = 600
mpl.rcParams["savefig.transparent"] = True
mpl.rcParams["font.size"] = 15
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
mpl.rcParams["axes.titlesize"] = "xx-large"  # medium, large, x-large, xx-large

mpl.style.use("seaborn-deep")

# %%
videopath = "/Genomics/ayroleslab2/scott/git/naps/docs/notebooks/example_1h.mp4"
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
aruco_params = cv2.aruco.DetectorParameters_create()

# %%
output_arr = np.empty((72000, 2, 100))
output_arr[:] = np.nan
cap = cv2.VideoCapture(videopath)
while cap.isOpened():

    ret, frame = cap.read()
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    # print(frame_number)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(
        frame, aruco_dict, parameters=aruco_params
    )
    print(len(corners))
    if len(corners) > 0:
        ids = ids.flatten()
    for (markerCorner, markerID) in zip(corners, ids):
        # extract the marker corners (which are always returned
        # in top-left, top-right, bottom-right, and bottom-left
        # order)
        # print(len(corners))
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        mid = (topLeft + bottomRight) / 2
        output_arr[frame_number, :, markerID] = mid
    if frame_number % 100 == 0:
        print(f"Frame number: {frame_number}")
    if frame_number == 72000:
        break

cap.release()
# cv2.destroyAllWindows()

# %%
mask = np.all(np.isnan(output_arr[:, 0, :]), axis=0)
cleaned_output = output_arr[:, :, ~mask]
cleaned_output = cleaned_output[:, np.newaxis, :, :]
cleaned_output.shape

# %%
np.save("output_aruco_1h.npy", cleaned_output)
