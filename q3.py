import cv2
import numpy as np
import matplotlib.pyplot as plt


def points_file_reader(file_name):
    """
    This function takes the name of the file containing coordinates of the points and returns the points as an array.
    :param file_name: name of the file containing coordinates of the points
    :return: a numeric array containing coordinates of the points
    """
    with open(file_name, 'r') as f:
        points_str = f.readlines()  # Reads all the lines at once
    # The first line is the number of points:
    n_points = points_str[0]
    # Remove the next line character
    n_points = int(n_points[:-1])
    # Separate coordinates by space and assign store them in a numpy array with shape = (n_points, dim)
    dim = len(points_str[2].split(' '))
    my_points = np.zeros((n_points, dim), dtype=int)
    points_str = points_str[1:]
    for i in range(n_points):
        point_i = points_str[i].split(' ')
        for j in range(dim):
            my_points[i, j] = float(point_i[j])

    return my_points


def add_contour(img, points):
    """
    this function draws a filled contour on the img
    :param img: the image
    :param points: the contour points
    :return: the image with the contour drawn on it
    """
    points_ = points.copy()
    # Don't need the image data!
    img_ = img.copy() * 0
    # Add the first point to the end so that the contour gets closed
    points_ = np.concatenate((points_, points_[0, :].reshape(1, 2)), axis=0)
    # Add the contour to the image
    img_ = cv2.drawContours(img_, [points_], 0, (255, 255, 255), thickness=-1)
    return img_


def event_handler(event, x, y, flags, param):
    """
    Event handler. Refer to q4.py of HW4
    :param event:
    :param x:
    :param y:
    :param flags:
    :param param:
    :return:
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append([y, x])
        print(x, y)
    else:
        pass


def get_user_points(input_image):
    """
    Gets the user initial contour. Refer to q4
    :param input_image: input image
    :return: user input points
    """
    user_points = []
    cv2.namedWindow("Source", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Source", event_handler, param=user_points)
    while True:
        cv2.imshow("Source", input_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    user_points = np.array(user_points)
    return user_points


source = cv2.imread("res08.jpg")
target = cv2.imread("res09.jpg")
# get_user_points(source)
source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

# print(source.shape)
# print(target.shape)
# Create an image with same size as the target. Put the source in this image
temp = np.zeros(target.shape, np.uint8)
temp[:source.shape[0], :source.shape[1]] = source
source = temp.copy()
# plt.imshow(source)
# plt.show()

# Read the bear (foreground-object) points
pts = points_file_reader("bear.txt")
# Create the binary mask for object
mask = add_contour(source, pts)
mask = mask == 255
mask = mask[:, :, 0] * mask[:, :, 1] * mask[:, :, 2]
mask = mask.astype(np.float64)
# plt.imshow(mask)
# plt.show()

h = source.shape[0]
w = source.shape[1]
# Shift the object to desired position
temp = np.concatenate((np.zeros((600, w)), mask), axis=0)
mask = temp[:h, :].copy()
temp = np.concatenate((np.zeros((600, w, 3), dtype=np.uint8), source), axis=0)
source = temp[:h, :, :].copy()
mask = np.concatenate((mask[:, 90:], np.zeros((h, 90))), axis=1)
source = np.concatenate((source[:, 90:, :], np.zeros((h, 90, 3))), axis=1)
# plt.imshow(source.astype(np.uint8))
# plt.show()
mask_ = np.zeros((mask.shape[0], mask.shape[1], 3), np.float64)
mask_[:, :, 0] = mask
mask_[:, :, 1] = mask
mask_[:, :, 2] = mask
mask = mask_

# Set the number of layers
n_levels = 5
source_gaussian_pyramid = []
source_laplacian_pyramid = []
target_gaussian_pyramid = []
target_laplacian_pyramid = []
# Add the images to the gaussian pyramid (stack)
# Note that the division by 255 isn't doing anything. My code had other bugs. But I kept it anyway(:
source_gaussian_pyramid.append(source.astype(np.float64) / 255)
target_gaussian_pyramid.append(target.astype(np.float64) / 255)

for i in range(1, n_levels):
    # Smooth the image in previous level of gaussian stack
    smoothed = cv2.GaussianBlur(source_gaussian_pyramid[i - 1], (121, 121), 20).astype(np.float64)
    # Deduce the smoothed image from the initial image to calculate the laplacian
    laplacian = source_gaussian_pyramid[i - 1].astype(np.float64) - smoothed.astype(np.float64)
    # Add the images to corresponding stack
    source_gaussian_pyramid.append(smoothed)
    source_laplacian_pyramid.append(laplacian)
    # Do the same for target image
    smoothed = cv2.GaussianBlur(target_gaussian_pyramid[i - 1], (121, 121), 20).astype(np.float64)
    laplacian = target_gaussian_pyramid[i - 1].astype(np.float64) - smoothed.astype(np.float64)
    target_gaussian_pyramid.append(smoothed)
    target_laplacian_pyramid.append(laplacian)

# Add the last layer of the gaussian stack to the laplacian stack
source_laplacian_pyramid.append(source_gaussian_pyramid[n_levels-1].astype(np.float64))
target_laplacian_pyramid.append(target_gaussian_pyramid[n_levels-1].astype(np.float64))
# print(len(source_laplacian_pyramid))
# print(len(target_laplacian_pyramid))

blended = []
for i in range(n_levels):
    # Blur the mask
    mask = cv2.GaussianBlur(mask, (121, 121), 20)
    # Do the blending
    blended.append(mask * source_laplacian_pyramid[i].astype(np.float64) + (1 - mask) * target_laplacian_pyramid[i].astype(np.float64))
    # plt.imshow(mask)
    # plt.show()
    # plt.imshow(blended[i])
    # plt.show()

out = np.zeros(source.shape, np.float64)
for i in range(n_levels):
    # Sum different layers
    out += blended[i]

# Scale back to 255
out = out * 255
# Clip the values out of [0, 255]
out[out > 255] = 255
out[out < 0] = 0
# Save the results
out = out.astype(np.uint8)
plt.imsave("res10.jpg", out)
# plt.imshow(out)
# plt.show()
