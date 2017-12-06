import random
import numpy as np
import cv2
class Image:
    # Default 1024x768
    def __init__(self, width=1024, height=768,two_d_array=[]):
        if len(two_d_array) >= 1:
            # print(two_d_array.shape)
            self.width = len(two_d_array[0])
            self.height = len(two_d_array)
            # There might be an issue here... come back if you have time
            # self.array = [[[0,0,0] for _ in range(height)] for _ in
            #  range(width)]
            # for w in range(width):
            #     for h in range(height):
            #         self.array[w][h] = two_d_array[h][w]
            # print(np.transpose(two_d_array, (1,0,2)).shape)
            self.array = np.transpose(two_d_array, (1,0,2))


        else:
            self.width = width
            self.height = height
            self.array = [[[random.randint(0,255),random.randint(0,255),random.randint(0,255)] for _ in range(height)] for _ in range(width)]

    def calculate_dist(self, image2):
        sum = 0
        for w in range(self.width):
            for h in range(self.height):
                pixel = self.pixel_at(w,h)
                pixel2 = image2.pixel_at(w,h)
                for i in range(len(pixel)):
                    sum += abs(pixel[i] -  pixel2[i])
        return sum
    def pixel_at(self,w, h):
        return self.array[w][h]
    def set_pixel(self,w,h,triple_array):
        self.array[w][h] = triple_array
    def apply_image(self,image2, weight):
        for w in range(self.width):
            for h in range(self.height):
                triple_array = self.pixel_at(w,h)
                triple_array2 = image2.pixel_at(w,h)
                for i in range(len(triple_array)):
                    self.array[w][h][i] = triple_array[i] * (1 - weight) + triple_array2[i] * weight