import random


class Image:
    # Default 1024x768
    def __init__(self, width=1024, height=768,two_d_array=[]):
        if len(two_d_array) >= 1:
            self.width = len(two_d_array[0])
            self.height = len(two_d_array)
            # print("width: " + str(self.width))
            # print("Height: " + str(self.height))
            self.array = two_d_array
        else:
            self.width = width
            self.height = height
            self.array = [[[random.randint(0,256),random.randint(0,256),random.randint(0,256)] for _ in range(width)] for _ in range(height)]

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
        return self.array[h][w]
    def set_pixel(self,w,h,triple_array):
        self.array[h][w] = triple_array
    def apply_image(self,image2, weight):
        for w in range(self.width):
            for h in range(self.height):
                triple_array = self.pixel_at(w,h)
                triple_array2 = image2.pixel_at(w,h)
                for i in range(len(triple_array)):
                    triple_array[i] = triple_array[i] * (1 - weight) + triple_array2[i] * weight