import cv2
from PIL import Image
import numpy as np

image = cv2.imread("./test-image-2.png", cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(image)
quantizedImage = pil_image.quantize(3)
quantizedImageCV2 = np.asarray(pil_image)
# convert to grayscale
gray = cv2.cvtColor(quantizedImageCV2, cv2.COLOR_BGR2GRAY)
(thresh, black_white) = cv2.threshold(gray, 126, 255, cv2.THRESH_BINARY)

# Getting white mask
mask = cv2.bitwise_not(black_white)

# Now we need to remove edges by making node mask, we erode heuristically until it stabilizes the amount of contours

kernel = np.ones((5, 5), np.uint8)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
numberOfContours = len(contours)

iteration = 0
contours_number_array = []

while (numberOfContours > 0):
    mask_eroded = cv2.erode(mask, kernel, iterations = iteration)
    contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    numberOfContours = len(contours)
    contours_number_array.append(numberOfContours)
    iteration += 1

ideal_contour_number = max(set(contours_number_array), key = contours_number_array.count)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
coutner_index = 0
iteration = 0
numberOfContours = len(contours)

while (numberOfContours != ideal_contour_number):
    mask_eroded = cv2.erode(mask, kernel, iterations = iteration)
    contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    numberOfContours = len(contours)
    if(numberOfContours == ideal_contour_number):
        coutner_index = iteration
    else:
        iteration += 1


# Made a mask for nodes
print(contours_number_array)
print(coutner_index)
graph_node = cv2.erode(mask, kernel, iterations = coutner_index)
# Need to dialiate to original size
graph_node = cv2.dilate(graph_node, kernel, iterations = coutner_index + 1)
graph_edges = mask - graph_node

# # Shows image
cv2.imshow("Example", mask)
cv2.imshow("Graph Node Mask", graph_node)
cv2.imshow("Graph Node Edges", graph_edges)


cv2.waitKey(0)
cv2.destroyAllWindows()

# Node labels

# for segmentation, we can also detect label regions within the nodes
# label detection is simple, findCOntours not on;ly provides outline of contour but all nested contours
# any nested contour marks a character inside the box, the bounding box of these contours builds the label region
# requires to contrast with bg, requires them to be inside nad the node needs to have solid fills
# we could alternatively use OCR


# Node detection
# nodes come in different shapes i.e rectangle, triangle, eclipse and color, outline stroke & stroke thickness

## Node shape detection
# pre-processing gives outline of the contour, deciding the shape
# we also want to detect if it's rotated or not

# openCV comes with a matchShapes() function with a similartity score of two given contours to be calculated
# function is a scale and translation invariant
# however it's rotation invariant which is why we're not using it

# what turned out to be a good approach, is creating specific shape mask with the same size for each serperate node region and the shape we want to detect
# with these mask, we calculate the difference between contour mask and these shape masks
# the similarityy then correlates with non-zero entries after bitwise XOR combinations of the inverted shape mask and te contour mask of the node itself
# the more non-zero entries in this bitwise the better reference mask matches the original contour

# Edge detection

# Edges exist as well, we need to find how edges connect and end
# later we'll try to find which node it cfonnects to

# we convert edge region which encompasses the entire path to something easier to work with
# skeletonization by morpological thinning
# skeleton retains basic shape and also stays connection, just reduced to 1 px wide. Making it easier to detect features

# Two end points are then matched to their closest node region to determine the two end nodes, furthermore since we have to end points, we know that the whole line is connected
# we can just walk along the neighboring pixels from one end point to the other to botain the edge path

# do to pruning step, we've pruned the actual end point
# we dialatte them until we find an intersection
