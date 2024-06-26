import cv2
from PIL import Image
import numpy as np
from skimage.morphology import skeletonize

import skyciv

def main():
    image = cv2.imread("./Test-2.png")

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
    (contours_array_type, contours_array_coordinate) = contour_identify_shape(graph_shapes)
    
    (graph_edges_parsed, endpoints_list) = find_edges_endpoints(graph_edges, graph_node)

    graph_edges_to_index = find_nodes(endpoints_list, contours_array_coordinate)

    # Prep blank image
    blank_image = np.zeros(image.shape, np.uint8)
        
    for i in range(0, len(graph_edges_to_index)):
        graph_edges_index = graph_edges_to_index[i]
        cv2.line(blank_image, contours_array_coordinate[graph_edges_index[0]], contours_array_coordinate[graph_edges_index[1]], color=(255, 255, 132), thickness=5)

    for i in range(0, len(contours_array_coordinate)):
        cv2.circle(blank_image, contours_array_coordinate[i], radius=2, color=(77,166,255), thickness=2)
        cv2.putText(blank_image, "Index: " + str(i), contours_array_coordinate[i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    # Shows image
    cv2.imshow("Temp", blank_image)
    cv2.imshow("Graph Node Edges", graph_edges)
    cv2.imshow("Graph Node Mask Shape", graph_shapes)
    cv2.imshow("Graph Edges After Parsing", graph_edges_parsed)


    cv2.waitKey(0)

    cv2.destroyAllWindows()

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
    contours_array_type = []

    for contour in contours:
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
            cv2.putText(graph_shapes, 'triangle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            contours_array_type.append("triangle")
    
        else:
            cv2.putText(graph_shapes, 'circle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            contours_array_type.append("circle")

        x,y,w,h = cv2.boundingRect(contour)
        contours_array.append((x, y))
    
    return (contours_array_type, contours_array)

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

def find_distance_two_points(pt1, pt2):
    x1,y1 = pt1
    x2, y2 = pt2
    dist = ((x2-x1)**2 + (y2-y1)**2 )**(1/2)
    return dist

def getMidPt(pairPnt):
    x1,y1 = pairPnt[0]
    x2,y2 = pairPnt[1]

    midPt = [int((x1+x2)/2), int((y1+y2)/2)]
    return midPt

def getEndpointsfromBox(box):
    boxList = box.tolist()
    dist_list = []
    for i, pt in enumerate(boxList):
        if i == 0:
            pt1 = pt
            pt2 = boxList[-1]
        else:
            pt1 = pt
            pt2 = boxList[i-1]
        
        dist = find_distance_two_points(pt1,pt2)
        dist_list.append(dist)
    
    ## get two least distances
    dist_list_copy  = dist_list.copy()
    dist_list_sorted = sorted(dist_list_copy)
    # print("dist_list",dist_list)
    # print("dist_list_sorted",dist_list_sorted)
    minD1, minD2 = dist_list_sorted[0], dist_list_sorted[1]
    index_minD1 = dist_list.index(minD1)
    index_minD2 = dist_list.index(minD2)
    
    if index_minD1 == index_minD2:
        indexList = [i for i,x in enumerate(dist_list) if x==minD1]
        index_minD1 = indexList[0]
        index_minD1 = indexList[1]

    
    if index_minD1 ==0:
        pair1 = boxList[0], boxList[-1]
    else:
        pair1 = boxList[index_minD1], boxList[index_minD1-1]
    
    if index_minD2 ==0:
        pair2 = boxList[0], boxList[-1]
    else:
        pair2 = boxList[index_minD2], boxList[index_minD2-1]

    endpt1 = getMidPt(pair1)
    endpt2 = getMidPt(pair2)
    return [endpt1, endpt2]

def find_edges_endpoints(graph_edges, graph_node):
    cv2.imwrite('temp_skeleton.png',graph_edges)
    skeleton_img_new = cv2.imread('temp_skeleton.png')
    
    cv2.imshow("skeleton_img",skeleton_img_new)
    # 2.2 Binarize the skeleton image

    skeleton_processed = skeletonize(skeleton_img_new)
    gray_skeleton = cv2.cvtColor(skeleton_processed, cv2.COLOR_BGR2GRAY)
    _, maskSkeleton = cv2.threshold(gray_skeleton, 1,255,0)
    
    edges_only = cv2.bitwise_or(maskSkeleton, graph_edges)

    ## WAY 2
    eroded = cv2.subtract(edges_only,graph_node)

    ## Process for contours
    contours_ed, heirarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
    h_img, w_img = graph_edges.shape[:2]
    newMask_ed = np.zeros((h_img, w_img,3),np.uint8)
    endpoint_list = []
    for cnt_ed in contours_ed:
        area = cv2.contourArea(cnt_ed)
        if area >8:
            # print('area',area)          
            newMask_ed = cv2.drawContours(newMask_ed, cnt_ed, -1,(0,255,0),1)
            rect = cv2.minAreaRect(cnt_ed)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            angle = rect[2]
            
            endpoints = getEndpointsfromBox(box)
            endpoint_list.append(endpoints)
            # newMask_ed = cv2.rectangle(newMask_ed,[box],(255,0,0),-1)
            newMask_ed = cv2.circle(newMask_ed,tuple(endpoints[0]),3,(0,0,255),-1)
            newMask_ed = cv2.circle(newMask_ed,tuple(endpoints[1]),3,(0,0,255),-1)
            # print(endpoints)
            # print(box)
    
    return (newMask_ed, endpoint_list)    

def find_closest_node(endpoint_coordinates, node_coords, index):
    node_coordinates = node_coords[:]
    
    if(index != -1):
        node_coordinates.pop(index)
    
    nodes = np.asarray(node_coordinates)
    dist_2 = np.sum((nodes - endpoint_coordinates)**2, axis=1)
    
    min = np.argmin(dist_2)

    if(index > min):
        return min + 1
    else:
        return min

def find_nodes(endpoints_list, contours_array_coordinate):
    graph_edges_to_index = []

    for i in range(0, len(endpoints_list)):
            # Can't be the same node
            closest_node_one = find_closest_node(endpoints_list[i][0], contours_array_coordinate, -1)
            closest_node_two = find_closest_node(endpoints_list[i][1], contours_array_coordinate, closest_node_one) + 1
            graph_edges_to_index.append((closest_node_one, closest_node_two))

    return graph_edges_to_index

if __name__ == "__main__":
    main()


