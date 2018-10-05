import cv2
import numpy as np


def draw_pts(image, gt_pts=None, pred_pts=None, get_l1e=False):
    l1_distances = []

    if gt_pts is not None:
        for i, pt in enumerate(pred_pts):
            if i == 90:
                break
            gpt = gt_pts[i]

            single_pt_distance = np.sqrt(np.square(pt[0] - gpt[0]) + np.square(pt[1] - gpt[1]))
            l1_distances.append(single_pt_distance)
            pred_pt = (int(pt[0]), int(pt[1]))
            grnd_pt = (int(gpt[0]), int(gpt[1]))
            cv2.circle(image, pred_pt, 2, (255, 0, 0))
            cv2.line(image, pred_pt, grnd_pt, (0, 0, 255))

        if get_l1e:
            return image, l1_distances
    else:
        for pt in pred_pts:
            pred_pt = (int(pt[0]), int(pt[1]))
            cv2.circle(image, pred_pt, 2, (255, 0, 0))
        return image



