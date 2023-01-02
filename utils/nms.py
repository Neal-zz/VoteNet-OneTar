import numpy as np

def nms_2d_faster(boxes, overlap_threshold):
    # boxes: K, 5
    x1 = boxes[:,0]     # minx
    y1 = boxes[:,1]     # minz
    x2 = boxes[:,2]     # maxx
    y2 = boxes[:,3]     # maxz
    score = boxes[:,4]  # probability
    area = (x2-x1)*(y2-y1)

    I = np.argsort(score)  # 从小到达存储 id，比如 [0,3,2,1]
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)  # add id

        xx1 = np.maximum(x1[i], x1[I[:last-1]])  # 令剩余元素的 minx>= x1[i]
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])

        # 相机坐标系下，xz 方向的最小包络盒
        w = np.maximum(0, xx2-xx1)
        h = np.maximum(0, yy2-yy1)

        inter = w*h
        o = inter / (area[i] + area[I[:last-1]] - inter)
        # 剔除重合度过大的盒子
        I = np.delete(I, np.concatenate(([last-1], np.where(o>overlap_threshold)[0])))

    return pick
