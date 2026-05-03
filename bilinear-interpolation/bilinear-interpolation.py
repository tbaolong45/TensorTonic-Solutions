import numpy as np
def bilinear_resize(image, new_h, new_w):
    image = np.array(image)
    
    h = image.shape[0]
    w = image.shape[1]

    image = np.pad(image, pad_width=((0, 1), (0, 1)), mode='constant', constant_values=0)
    
    if new_w == 1:
        c_x = 0
    else:
        c_x = (w - 1) / (new_w - 1)
        
    if new_h == 1:
        c_y = 0
    else:
        c_y = (h - 1) / (new_h - 1)

    dx = 0
    dy = 0

    pos_x = 0
    pos_y = 0

    res = []
    
    for i in range(new_h):
        tmp = []
        
        for j in range(new_w):

            p_x = int(pos_x)
            p_y = int(pos_y)
            
            val = (1 - dy) * (1 - dx) * image[p_y][p_x] + dx * (1 - dy) * image[p_y][p_x + 1] + (1 - dx) * dy * image[p_y + 1][p_x] + dy * dx * image[p_y + 1][p_x + 1]

            tmp.append(val)

            # print(image[p_x][p_y], image[p_x][p_y + 1], image[p_x + 1][p_y], image[p_x + 1][p_y + 1])
            print(dx, dy)
            pos_x += c_x
            dx = (dx + c_x) % 1
            

        res.append(tmp)

        pos_x = 0
        pos_y += c_y

        dx = 0
        dy = (dy + c_y) % 1

    return list(map(lambda x: list(map(float, x)), res))