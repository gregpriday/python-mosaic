import cv2
import random
import numpy as np

class VoronoiMosaic(object):
    def __init__(self):
        self.ref_image = None
        self.vor_image = None
        self.point_x = []
        self.point_y = []

    def resize_image(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    def preprocess_image(self, image_file):
        # Load image and store as array (convert to RGB)
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize it down to 512x512 pixels
        img = self.resize_image(image=img, height=512)

        # Apply gaussian blur
        img = cv2.GaussianBlur(img, (10, 10), 0)

        # quantize a image
        img = img.quantize(256)

        self.ref_image = img

    def generate_seeds(self):
        # grab the image size
        (h, w) = self.ref_image.shape[:2]

        self.point_x = random.sample(range(0, w), 512)
        self.point_y = random.sample(range(0, h), 512)

    def initialize_dic(self, size):
        dic = {}
        for i in range(size):
            # We push data pair (-1, -1) into dic for every point
            # It will be removed while processing something
            dic[i] = np.array([[-1], [-1]])
        return dic

    def update_positions(self, dic, point_x, point_y):
        for i in range(point_x.size):
            # Get all the pixels belongs to the cell and remove first dummy data pair (-1, -1)
            data = dic[i][:, 1:]
            # Avoid that cells contain nothing but still process
            if (data.size > 0):
                new_px = round(np.average(data[0]))
                new_py = round(np.average(data[1]))
                point_x[i] = new_px
                point_y[i] = new_py

    def voronoi_diagram(self, width, height, point_x, point_y, dic):
        # For each pixel
        for y in range(height):
            for x in range(width):
                # Compute difference (position of cell - position of pixel)
                diff = np.array([point_x - x, point_y - y])
                # Calculate Euclidean distance
                dist = np.linalg.norm(diff, axis=0)
                # Find the smallest index inside the numpy array.
                # That is, the index stands for the cell (or point)
                index = np.argmin(dist)
                # Make new data pair
                coord = np.array([[x], [y]])
                # Put the new data pair inside the dictionary with its key is the point
                dic[index] = np.hstack((dic[index], coord))

    def generate_voronoi_diagram(self, width, height, point_x, point_y, iteration=3):
        for i in range(iteration):
            dic = self.initialize_dic(point_x.size)
            self.voronoi_diagram(width, height, point_x, point_y, dic)
            self.update_positions(dic, point_x, point_y)
        return dic

    def generate_voronio_image(self, img, points_x, points_y, dic):
        # Create an image buffer
        output_img = np.full(img.size, 255).reshape(img.shape).astype(np.uint8)

        # For every cell
        for i in range(points_x.size):
            x = int(round(points_x[i]))
            y = int(round(points_y[i]))
            data = dic[i][:, 1:]
            # Retrieve rbg value of raw image
            r = int(img[y, x, 0])
            g = int(img[y, x, 1])
            b = int(img[y, x, 2])

            # For every pixel in the cell
            for j in range(data[0].size):
                tx = data[0][j]
                ty = data[1][j]
                r += img[ty, tx, 0]
                g += img[ty, tx, 1]
                b += img[ty, tx, 2]

            # Compute average rgb value
            r /= (data[0].size + 1)
            g /= (data[0].size + 1)
            b /= (data[0].size + 1)

            # Copy rbg value to centroidal point
            output_img[y, x, :] = np.array([r, g, b])
            # Copy rbg value to every pixel
            for j in range(data[0].size):
                tx = data[0][j]
                ty = data[1][j]
                output_img[ty, tx, :] = np.array([r, g, b])

        self.vor_image = output_img
        return output_img

    def calc_e_color(self):
        # Compute difference
        color_diff = np.array([self.ref_image - self.vor_image])
        # Calculate Euclidean distance
        e_color = np.linalg.norm(color_diff) ** 2
        return e_color