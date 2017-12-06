import math
import Image
import numpy as np


class SOM:
    # Default image size 1024x768
    def __init__(self, width, height, image_width, image_height, radius, learning_rate=0.9, decay_rate=0.9):
        self.width = width
        self.height = height
        self.radius = radius
        self.learning_rate = learning_rate
        self.image_width = image_width
        self.image_height = image_height
        self.SOMArray = [[Image.Image(image_width,image_height) for j in range(height)] for i in range(width)]
        self.weight_array = [[learning_rate for _ in range(height)] for _ in range(width)]
        self.decay_rate = decay_rate
        self.been_trained = False

    # Pass in the observation_n[0]["vision"] array
    def train(self, state_array):
        state_image = Image.Image(two_d_array=state_array)
        if not self.been_trained:
            self.been_trained = True
            best_height = int(self.height / 2)
            best_width = int(self.width / 2)
            self.get(best_height,best_width)
            return best_width, best_height
        else:
            best_w, best_h = self.find_coord(state_image)
            neighbor_triples = self.get_neighbors(best_w,best_h)
            for neighbor_triple in neighbor_triples:
                ##  triple of (w, h, rate)
                self.apply_array_to_coordinate(state_image,neighbor_triple[0],neighbor_triple[1],neighbor_triple[2])
            return best_w, best_h
    ## Applies the array to the neighbors
    def apply_array_to_coordinate(self, new_image, w, h, weight):
        image = self.get(w,h)
        image.apply_image(new_image,weight)
        self.weight_array[w][h] *= self.decay_rate

    ## Returns a triple of neighbors (w, h, rate)
    def get_neighbors(self, w, h):
        neighbors = []
        for i in range(self.radius):
            rate = (self.learning_rate / (i + 1)) * self.weight_array[w][h]
            neighbors += self.get_neighbors_x_away(w,h,i + 1,rate)
        return neighbors

    def get_neighbors_x_away(self, w, h, x, weight):
        result = []
        if w + x < self.width:
            result.append((w+x,h,weight))
        if w - x >= 0:
            result.append(((w-x),h,weight))
        if h + x < self.height:
            result.append((w,(h+x),weight))
        if h - x >= 0:
            result.append((w,(h - x),weight))
        return result
    ## Returns the coordinates of the closest SOM square
    def find_coord(self, image2):
        min_dist = float("inf")
        best_h = 0
        best_w = 0
        for h in range(self.height):
            for w in range(self.width):
                diff = self.get(w,h).calculate_dist(image2)
                if diff < min_dist:
                    min_dist = diff
                    best_h = h
                    best_w = w
        return best_w, best_h

    def set(self, height, width, array):
        self.SOMArray[width][height] = array

    def get(self, width, height):
        return self.SOMArray[width][height]
    # def SOM_as_giant_array(self):
    #     resulting_array = [[[0,0,0] for column in range(self.width * self.image_width)] for row in range(self.height * self.image_height)]
    #     for row in range(len(resulting_array)):
    #         for column in range(len(resulting_array[0])):
    #             gridy = row // self.image_height
    #             gridx = column // self.image_width
    #             pixely = row % self.image_height
    #             pixelx = column % self.image_width
    #             resulting_array[row][column] = int(self.get(gridy,gridx).pixel_at(pixelx,pixely))
    #     return np.array(resulting_array)

## This assumes that we are getting an RGB grid.
def calculate_difference(array1, array2):
    sum = 0
    maxval = 10 # arbitrary number max
    for i in range(len(array1)):
        if math.isinf(array1[i]) or math.isinf(array2[i]):
            sum += maxval
        else:
            sum += abs(array1[i] - array2[i])
    return sum