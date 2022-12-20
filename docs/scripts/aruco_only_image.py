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
image = cv2.imread(
    "/Genomics/ayroleslab2/scott/git/naps/docs/scripts/fig1_bees.png", cv2.IMREAD_COLOR
)

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
aruco_params = cv2.aruco.DetectorParameters_create()

# %%
output_arr = np.empty((1, 2, 100))
output_arr[:] = np.nan
image_number = 0


# print(image_number)
(corners, ids, rejected) = cv2.aruco.detectMarkers(
    image, aruco_dict, parameters=aruco_params
)


if len(corners) > 0:
    # flatten the ArUco IDs list
    ids = ids.flatten()
    # loop over the detected ArUCo corners
    for (markerCorner, markerID) in zip(corners, ids):
        # extract the marker corners (which are always returned in
        # top-left, top-right, bottom-right, and bottom-left order)
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        # convert each of the (x, y)-coordinate pairs to integers
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))
        # draw the bounding box of the ArUCo detection
        cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
        # compute and draw the center (x, y)-coordinates of the ArUco
        # marker
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
        # draw the ArUco marker ID on the image
        cv2.putText(
            image,
            str(markerID),
            (topLeft[0], topLeft[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        print("[INFO] ArUco marker ID: {}".format(markerID))
        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)

# cv2.destroyAllWindows()

# %%
mask = np.all(np.isnan(output_arr[:, 0, :]), axis=0)
cleaned_output = output_arr[:, :, ~mask]
cleaned_output = cleaned_output[:, np.newaxis, :, :]
cleaned_output.shape

# %%
np.save("output_aruco_1h.npy", cleaned_output)
