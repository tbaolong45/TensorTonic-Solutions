import math

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    stride = int(image_size / feature_size)

    tmp = []
    
    for i in range(feature_size):
        for j in range(feature_size):
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride

            for s in scales:
                for r in aspect_ratios:
                    b_W = s * math.sqrt(r)
                    b_H = s / math.sqrt(r)
                    
                    tmp.append([cx - b_W/2, cy - b_H/2, cx + b_W/2, cy + b_H/2])

    return tmp
            