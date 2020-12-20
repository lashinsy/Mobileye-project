from math import sqrt
import numpy as np
from phase_III.frame_container import FrameContainer


def calc_tfl_dist(prev_container: FrameContainer, curr_container: FrameContainer, focal: float, pp: np.array) \
        -> FrameContainer:
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if abs(tZ) < 10e-6:
        print('tz = ', tZ)
    elif norm_prev_pts.size == 0:
        print('no prev points')
    elif norm_prev_pts.size == 0:
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, \
        curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)

    return curr_container


def prepare_3D_data(prev_container: FrameContainer, curr_container: FrameContainer, focal: float, pp: np.array) \
        -> (np.ndarray, np.ndarray, np.ndarray, np.array, float):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(curr_container.EM)

    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts: np.ndarray, norm_curr_pts: np.ndarray, R: np.ndarray, foe: np.array, tZ: float) \
        -> (list, np.array, list):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []

    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)

    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts: np.array, focal: float, pp: np.array) -> np.ndarray:
    # transform pixels into normalized pixels using the focal length and principle point
    return (pts - pp) / focal


def unnormalize(pts: np.array, focal: float, pp: np.array) -> np.ndarray:
    # transform normalized pixels into pixels using the focal length and principle point
    return pts * int(focal) + pp


def decompose(EM: np.ndarray) -> (np.ndarray, float):
    t = EM[:3, 3]

    return EM[:3, :3], np.array([t[0], t[1]]) / t[2], t[2]


def get_EM(curr: int, prev: int, data):
    EM = np.eye(4)

    for i in range(prev, curr):
        EM = np.dot(data['egomotion_' + str(i) + '-' + str(i + 1)], EM)

    return EM


def rotate(pts: np.ndarray, R: np.ndarray) -> [np.array]:
    res = [np.dot(R, np.append(p, 1)) for p in pts]

    return [(p / p[2])[:2] for p in res]


def find_corresponding_points(p: np.array, norm_pts_rot: list, foe: np.array) -> (int, np.array):
    # compute the epipolar line between p and foe
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = (p[1] * foe[0] - foe[1] * p[0]) / (foe[0] - p[0])
    # run over all norm_pts_rot and find the one closest to the epipolar line
    distance = [abs((m * p[0] + n - p[1]) / sqrt(m ** 2 + 1)) for p in norm_pts_rot]
    min_point = distance.index(min(distance))
    # return the closest point and its index
    return min_point, norm_pts_rot[min_point]


def calc_dist(p_curr: np.array, p_rot: np.array, foe: np.array, tZ: float) -> float:
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    Zx = tZ * (foe[0] - p_rot[0]) / (p_curr[0] - p_rot[0])
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    Zy = tZ * (foe[1] - p_rot[1]) / (p_curr[1] - p_rot[1])
    # combine the two estimations and return estimated Z
    dx = abs(p_rot[0] - p_curr[0])
    dy = abs(p_rot[1] - p_curr[1])

    return abs(Zx * dx / (dx + dy) + Zy * dy / (dx + dy))
