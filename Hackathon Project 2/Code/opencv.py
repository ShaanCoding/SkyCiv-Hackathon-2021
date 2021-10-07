import cv2
from PIL import Image
import numpy as np

def main():
    image = cv2.imread("./test-image-3.png")

    # Preprocessing
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

    (coutner_index) = find_contour_index(mask, kernel)

    # Made a mask for nodes
    graph_node = cv2.erode(mask, kernel, iterations = coutner_index)
    # Need to dialiate to original size
    graph_node = cv2.dilate(graph_node, kernel, iterations = coutner_index + 1)
    graph_edges = mask - graph_node
    
    graph_shapes = graph_node.copy()
    (contours, contours_array_type, contours_array_coordinate) = contour_identify_shape(graph_shapes)

    # for i in range(0, len(contours_array_type)):
        # print(contours_array_type[i] + " " + np.array2string(contours_array_coordinate[i]))

    graph_edges_skeleton = morpological_skeletonization(graph_edges)

    coordinates = find_intersecting_contours(black_white, graph_node, graph_edges_skeleton)
    print(coordinates)

    # skeleton_coordinates = convert_skeleton_to_coordinates(graph_edges_skeleton)


    # Shows image
    cv2.imshow("Example", mask)
    cv2.imshow("Graph Node Mask", graph_node)
    
    cv2.imshow("Graph Node Edges", graph_edges)
    cv2.imshow("Graph Node Skeleton", graph_edges_skeleton)



    cv2.imshow("Graph Node Mask Shape", graph_shapes)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# Edge detection

# Edges exist as well, we need to find how edges connect and end
# later we'll try to find which node it cfonnects to

# Two end points are then matched to their closest node region to determine the two end nodes, furthermore since we have to end points, we know that the whole line is connected
# we can just walk along the neighboring pixels from one end point to the other to botain the edge path

# do to pruning step, we've pruned the actual end point
# we dialatte them until we find an intersection

def find_contour_index(mask, kernel):
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
    
    return coutner_index

def contour_identify_shape(graph_shapes):
    i = 0
    contours, hierarchy = cv2.findContours(graph_shapes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_array = []

    for contour in contours:
        if i == 0:
            i = 1
            continue

        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        
        cv2.drawContours(graph_shapes, [contour], 0, (0, 0, 255), 5)
    
        # finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
    
        # putting shape name at center of each shape
        if len(approx) >= 3 and len(approx) <= 6:
            cv2.putText(graph_shapes, 'Triangle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            contours_array.append("Triangle")
    
        else:
            cv2.putText(graph_shapes, 'circle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            contours_array.append("Circle")
    
    return (contours, contours_array, np.vstack(contours).squeeze())

def morpological_skeletonization(graph_edges):
    img = graph_edges.copy()

    size = np.size(img)
    skeleton = np.zeros(img.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    done = False

    while(not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if (zeros == size):
            done = True

    return skeleton

# My implementation of attempting to find contour coordiantes
def convert_skeleton_to_coordinates(skeleton):
    # https://stackoverflow.com/questions/26537313/how-can-i-find-endpoints-of-binary-skeleton-image-in-opencv

    img = skeleton.copy()

    (rows, cols) = np.nonzero(img)

    # Initialize empty list of coordinates
    skel_coords = []

    # For each non-zero pixel
    for (r,c) in zip(rows, cols):
        # Extract an 8-connected neighbourhood
        (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))

        # Cast to int to index into image
        col_neigh = col_neigh.astype('int')
        row_neigh = row_neigh.astype('int')

        # Convert into a single 1D array and check for non-zero locations
        pix_neighbourhood = img[row_neigh,col_neigh].ravel() != 0

        # If the number of non-zero locations equals 2, add this to 
        # our list of co-ordinates
        if np.sum(pix_neighbourhood) == 2:
            skel_coords.append((r,c))
    return skel_coords

def find_intersecting_contours(original_image, graph_node, graph_edges_skeleton):
    contours_graph_node, hierarchy_graph_node = cv2.findContours(graph_node, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_graph_edges, hierarchy_graph_edges = cv2.findContours(graph_edges_skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = [contours_graph_node, contours_graph_edges]

    # Create image filled with zeros
    zeroes = np.zeros(original_image.shape[0:2])

    # Copy each contour into it's own image and fill it with '1'
    image1 = cv2.drawContours(zeroes.copy(), contours[0], 0, 1)
    image2 = cv2.drawContours(zeroes.copy(), contours[1], 1, 1)

    # Use the logical AND operator on two images
    intersection = np.logical_and(image1, image2)

    # Check if there was a '1' in the intersection
    return intersection.any()
# End of implementation

def list_ports():
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports

if __name__ == "__main__":
    main()

