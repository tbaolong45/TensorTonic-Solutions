import math
import numpy as np

def rotate_image(image, angle_degrees):
    image = np.array(image)

    cx = (image.shape[1] - 1) / 2
    cy = (image.shape[0] - 1) / 2

    res = []
    
    angle_degrees = math.radians(angle_degrees)

    for i in range(image.shape[0]):
        tmp = []
        for j in range(image.shape[1]):
            dy = i - cy
            dx = j - cx

            src_x = cx - dy * math.sin(angle_degrees) + dx * math.cos(angle_degrees)
            src_y = cy + dy * math.cos(angle_degrees) + dx * math.sin(angle_degrees)

            if round(src_x) < 0 or round(src_x) >= image.shape[1] or round(src_y) < 0 or round(src_y) >= image.shape[0]:
                tmp.append(0)
                continue

            print(src_y, src_x)

            src_x = round(src_x)
            src_y = round(src_y)


            tmp.append(image[src_y][src_x])
        res.append(tmp)
    return [list(map(float, row)) for row in res]