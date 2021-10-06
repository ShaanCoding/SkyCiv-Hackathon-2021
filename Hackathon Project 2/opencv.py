import cv2

image = cv2.imread("./test-image.png", cv2.IMREAD_COLOR)

cv2.imshow("Example", image)

cv2.waitKey(0)

cv2.destroyAllWindows()

# Break down to either node or edge
# Can be processed independently to detect different features (e.g. color, shape, edge bends, etc)

# Breaking it apart

# Seperate it into nodes & edge
# Simplifies it, this is important but there's no main way to split them

# Morphology based segmentation

# Change to binary
# Grayscale to B/W
# outermost contour in the image and assume this is the foreground (i.e graph) contour is inital mask

# Then we remove edghes by running multiple erode bypasses
# After multiple erosions (which trims a big feature) nodes will shrink whilst edges will be deleted
# Once done, we run opencv findContours() to detect node
# Given node contours, we can create an edge mask which can detect our edge regions of our segment

# how to erode imageu until all edges are removed?
# depends on thickness of edge, heuristic
# at some point edges will dissapear and nodes now form their own contours

# bascially we repeat until contours number stablizes and we get the min number of eroisions

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
