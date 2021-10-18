# SkyCiv Structural Engineering Hackathon

- This program works using Python, Numpy & OpenCV

This works by:

- Taking in an input image
- Converting it into a 3 color image
- Converting it to GRAYSCALE
- Binarizing the GRAYSCALE image to B/W

- After doing this we get a mask of all "Nodes" by eroding the binarized image until we reach 0 contours
- We then find the most common number of contours from all iterations based on frequency

- After this we find the edges by subtracting the node mask mask from the original binarized file

- From this we have the nodes and the edges identified

- We then identify each node by using image processing; labelling it as either a circle, parallelogram & triangle
