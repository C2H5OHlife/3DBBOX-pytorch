import numpy as np
from wavedata.tools.obj_detection.obj_utils import *
from wavedata.tools.visualization.vis_utils import *
import matplotlib.pyplot as plt


all_case = [
        # 斜 + 俯
        [[0], [2], [4, 5, 6], [0, 2, 3]],  # [0, 2, 5, 3]
        [[1], [3], [5, 6, 7], [0, 1, 3]],  # [1, 3, 6, 0]
        [[2], [0], [4, 6, 7], [0, 1, 2]],  # [2, 0, 7, 1]
        [[3], [1], [4, 5, 7], [1, 2, 3]],  # [3, 1, 4, 2]

        # 正 + 俯
        [[0], [3], [5, 6], [0, 3]],  # [0 ,3, 5, 3]
        [[1], [0], [6, 7], [0, 1]],  # [1, 0, 6, 0]
        [[2], [1], [4, 7], [1, 2]],  # [2, 1, 7, 1]
        [[3], [2], [4, 5], [2, 3]],  # [3, 0, 4, 2]

        # 正 + 平(仰)
        [[0], [3], [4, 7], [0, 1, 2, 3]],  # [0, 3, 4, 3]
        [[1], [0], [4, 5], [0, 1, 2, 3]],  # [1, 0, 5, 0]
        [[2], [1], [5, 6], [1, 2, 2, 3]],  # [2, 1, 6, 1]
        [[3], [2], [6, 7], [0, 1, 2, 3]],  # [3, 2, 7, 2]

        # 侧 + 平（仰）
        [[0], [2], [4, 6, 7], [0, 1, 2, 3]],  # [0, 2, 7, 3]
        [[1], [3], [4, 5, 7], [0, 1, 2, 3]],  # [1, 3, 4, 0]
        [[2], [0], [4, 5, 6], [0, 1, 2, 3]],  # [2, 0, 5, 1]
        [[3], [1], [5, 6, 7], [0, 1, 2, 3]],  # [3, 1, 6, 2]

        # # 斜 + 仰
        # [[0], [2], [7, ], [1, ]],
        # [[1], [3], [4, ], [2, ]],
        # [[2], [0], [5, ], [3, ]],
        # [[3], [1], [6, ], [0, ]],
        #
        # # 正 + 仰
        # [[0], [3], [4], [1]],
        # [[1], [0], [5], [2]],
        # [[2], [1], [6], [3]],
        # [[3], [2], [7], [0]],

        # # 斜 + 俯 2.0
        # [[0], [2], [6], [0]],
        # [1, 3, 7, 1],
        # [2, 4, 4, 2],
        # [3, 5, 5, 3],
        #
        # # 平 + 侧 2.0
        # [0, 2, 6, 2],
        # [1, 3, 7, 3],
        # [2, 0, 4, 0],
        # [3, 1, 5, 1],
        #
        # # 正 + 平 2.0
        # [0, 3, 4, 0],
        # [1, 0, 5, 1],
        # [2, 1, 6, 2],
        # [3, 2, 7, 3],  # 7->6
        #
        # # 正 + 俯
        # [0, 3, 6, 0],  #
        # [1, 0, 7, 1],
        # [2, 1, 4, 2],
        # [3, 2, 5, 3],
    ]

all_case_exclude = [
# 斜 + 俯
        [[0, 2, 5, 2], [0, 2, 4, 0]],
        [[1, 3, 7, 3], [1, 3, 5, 1]],
        [[2, 0, 4, 0], [2, 0 ,6, 2]],
        [[3, 1, 5, 1], [3, 1, 7, 3]],

        # 正 + 俯
        [[]],
        [[]],
        [[]],
        [[]],

        # 正 + 平(仰)
        [[]],  # [0, 3, 4, 3]
        [[]],  # [1, 0, 5, 0]
        [[]],  # [2, 1, 6, 1]
        [[]],  # [3, 2, 7, 2]

        # 侧 + 平（仰）
        [[]],  # [0, 2, 7, 3]
        [[]],  # [1, 3, 4, 0]
        [[]],  # [2, 0, 5, 1]
        [[]],  # [3, 1, 6, 2]
]


all_case_std = [
        # 斜 + 俯
        [0, 2, 5, 3],
        [1, 3, 6, 0],
        [2, 0, 7, 1],
        [3, 1, 4, 2],

        # 正 + 俯
        [0 ,3, 5, 3],
        [1, 0, 6, 0],
        [2, 1, 7, 1],
        [3, 0, 4, 2],

        # 正 + 平(仰)
        [0, 3, 4, 3],
        [1, 0, 5, 0],
        [2, 1, 6, 1],
        [3, 2, 7, 2],

        # 侧 + 平（仰）
        [0, 2, 7, 3],
        [1, 3, 4, 0],
        [2, 0, 5, 1],
        [3, 1, 5, 1],  # [3, 1, 6, 2],
    ]


def viewpoint(d, t, ry, p):

    left = 0
    right = 0
    up = 0
    down = 0

    rot = np.array([[+np.cos(ry), 0, +np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, +np.cos(ry)]])

    l = d[2]
    w = d[1]
    h = d[0]

    # 3D BB corners
    x_corners = np.array(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    corners_3d = np.dot(rot, np.array([x_corners, y_corners, z_corners]))

    corners_3d[0, :] = corners_3d[0, :] + t[0]
    corners_3d[1, :] = corners_3d[1, :] + t[1]
    corners_3d[2, :] = corners_3d[2, :] + t[2]

    cube, face_idx = project_box3d_to_image(corners_3d, p)

    min_x, max_x = cube[0, 0], cube[0, 0]
    min_y, max_y = cube[1, 0], cube[1, 0]

    for i in range(1, 8):
        if min_x > cube[0, i]:
            min_x = cube[0, i]
            left = i
        # elif min_x == cube[0, i]:
        #     left.add(i)
        if max_x < cube[0, i]:
            max_x = cube[0, i]
            right = i
        # elif max_x == cube[0, i]:
        #     right.add(i)
        if min_y > cube[1, i]:
            min_y = cube[1, i]
            up = i
        # elif min_x == cube[1, i]:
        #     up.add(i)
        if max_y < cube[1, i]:
            max_y = cube[1, i]
            down = i
        # elif max_y == cube[1, i]:
        #     down.add(i)

    # flag = 1
    # print(left)
    # print(right)
    # print(up)
    # print(down)
    for i, case in enumerate(all_case):
        if left in case[0] and right in case[1] and up in case[2] and down in case[3]:
            # print('case ' + str(i))
            # flag = 0
            if [left, right, up, down] in all_case_exclude[i]: continue
            return i
    return -1
    # if flag and d[0] != -1.0:
    #     print('case not found:')
    #     ax = plt.axes()
    #     # cube = np.concatenate((cube, np.ones((1, 8),dtype=np.float64)), axis=0)
    #     plot_3d(cube, ax)
    #     plt.draw()
    #     ax.set_ylim(ax.get_ylim()[::-1])
    #     plt.pause(0.5)
    # return left, right, up, down


def plot_3d(corners, ax, c='lime'):
    """Plots 3D cube

    Arguments:
        corners: Bounding box corners
        ax: graphics handler
    """

    # Draw each line of the cube
    p1 = corners[:, 0]
    p2 = corners[:, 1]
    p3 = corners[:, 2]
    p4 = corners[:, 3]

    p5 = corners[:, 4]
    p6 = corners[:, 5]
    p7 = corners[:, 6]
    p8 = corners[:, 7]

    #############################
    # Bottom Face
    #############################
    ax.plot([p1[0], p2[0]],
            [p1[1], p2[1]],
            c='red')

    ax.plot([p2[0], p3[0]],
            [p2[1], p3[1]],
            c=c)

    ax.plot([p3[0], p4[0]],
            [p3[1], p4[1]],
            c=c)

    ax.plot([p4[0], p1[0]],
            [p4[1], p1[1]],
            c='red')

    #############################
    # Top Face
    #############################
    ax.plot([p5[0], p6[0]],
            [p5[1], p6[1]],
            c=c)

    ax.plot([p6[0], p7[0]],
            [p6[1], p7[1]],
            c=c)

    ax.plot([p7[0], p8[0]],
            [p7[1], p8[1]],
            c=c)

    ax.plot([p8[0], p5[0]],
            [p8[1], p5[1]],
            c=c)
    #############################
    # Front-Back Face
    #############################
    ax.plot([p5[0], p8[0]],
            [p5[1], p8[1]],
            c=c)

    ax.plot([p8[0], p4[0]],
            [p8[1], p4[1]],
            c=c)

    ax.plot([p4[0], p1[0]],
            [p4[1], p1[1]],
            c='red')

    ax.plot([p1[0], p5[0]],
            [p1[1], p5[1]],
            c='red')
    #############################
    # Front Face
    #############################
    ax.plot([p2[0], p3[0]],
            [p2[1], p3[1]],
            c=c)

    ax.plot([p3[0], p7[0]],
            [p3[1], p7[1]],
            c=c)

    ax.plot([p7[0], p6[0]],
            [p7[1], p6[1]],
            c=c)

    ax.plot([p6[0], p2[0]],
            [p6[1], p2[1]],
            c=c)


def compute_location(box_2d, d, ry, p, v):
    """
    用二维框位置，三维框大小，全局角度和内参矩阵计算三维位置t
    :param box_2d: 维框位置
    :param d: 三维框大小
    :param ry: 全局弧度
    :param p: 内参矩阵
    :return: t 目标三维位置
    """
    (x_min, y_min), (x_max, y_max) = box_2d
    # inv_p = np.linalg.inv(p[:, 0:2])
    rot = np.array([[+np.cos(ry), 0, +np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, +np.cos(ry)]])

    l = d[2]
    w = d[1]
    h = d[0]

    # 3D BB corners
    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    corners_3d = np.dot(rot, np.array([x_corners, y_corners, z_corners]))


    cons_0 = np.dot(p, np.append(corners_3d[:, all_case_std[v][0]].reshape(3, 1), np.ones(shape=(1, 1)), axis=0))
    cons_0 = (cons_0[2, 0] + p[2, 3]) * x_min - cons_0[0, 0] - p[0, 3]
    cons_1 = np.dot(p, np.append(corners_3d[:, all_case_std[v][1]].reshape(3, 1), np.ones(shape=(1, 1)), axis=0))
    cons_1 = (cons_1[2, 0] + p[2, 3]) * x_max - cons_1[0, 0] - p[0, 3]
    cons_2 = np.dot(p, np.append(corners_3d[:, all_case_std[v][2]].reshape(3, 1), np.ones(shape=(1, 1)), axis=0))
    cons_2 = (cons_2[2, 0] + p[2, 3]) * y_min - cons_2[1, 0] - p[2, 3]
    cons_3 = np.dot(p, np.append(corners_3d[:, all_case_std[v][3]].reshape(3, 1), np.ones(shape=(1, 1)), axis=0))
    cons_3 = (cons_3[2, 0] + p[2, 3]) * y_max - cons_3[1, 0] - p[2, 3]
    P_mat = np.concatenate((p[0, 0:3] - x_min * p[2, 0:3],
                            p[0, 0:3] - x_max * p[2, 0:3],
                            p[1, 0:3] - y_min * p[2, 0:3],
                            p[1, 0:3] - y_max * p[2, 0:3]), axis=0).reshape(4, 3)
    cons = np.array([[cons_0], [cons_1], [cons_2], [cons_3]])
    T = np.linalg.solve(np.dot(P_mat.T, P_mat), np.dot(P_mat.T, cons))

    # cons_0 = corners_3d[2, all_case_std[v][0]] * x_min - np.dot(p, np.append(corners_3d[:, all_case_std[v][0]].reshape(3, 1), np.ones(shape=(1, 1)), axis=0))[0, 0] - p[0, 3]
    # cons_1 = corners_3d[2, all_case_std[v][1]] * x_max - np.dot(p, np.append(corners_3d[:, all_case_std[v][1]].reshape(3, 1), np.ones(shape=(1, 1)), axis=0))[0, 0] - p[0, 3]
    # cons_2 = corners_3d[2, all_case_std[v][2]] * y_min - np.dot(p, np.append(corners_3d[:, all_case_std[v][2]].reshape(3, 1), np.ones(shape=(1, 1)), axis=0))[1, 0] - p[1, 3]
    # cons_3 = corners_3d[2, all_case_std[v][3]] * y_max - np.dot(p, np.append(corners_3d[:, all_case_std[v][3]].reshape(3, 1), np.ones(shape=(1, 1)), axis=0))[1, 0] - p[1, 3]
    # cons = np.array([[cons_0], [cons_1], [cons_2], [cons_3]])
    # P_mat = np.concatenate((p[0, 0:3], p[0, 0:3], p[1, 0:3], p[1, 0:3]), axis=0).reshape(4, 3)
    # T = np.linalg.solve(np.dot(P_mat.T, P_mat), np.dot(P_mat.T, cons))
    # cons_0 = x_min - np.dot(p[: ,0:3], corners_3d[:, all_case_std[v][0]].reshape(3, 1))[0, 0] - p[0, 3]
    # cons_1 = x_max - np.dot(p[:, 0:3], corners_3d[:, all_case_std[v][1]].reshape(3, 1))[0, 0] - p[0, 3]
    # cons_2 = y_min - np.dot(p[:, 0:3], corners_3d[:, all_case_std[v][2]].reshape(3, 1))[1, 0] - p[1, 3]
    # cons_3 = y_max - np.dot(p[:, 0:3], corners_3d[:, all_case_std[v][3]].reshape(3, 1))[1, 0] - p[1, 3]
    # cons = np.array([[cons_0], [cons_1], [cons_2], [cons_3]])
    # P_mat = np.concatenate((p[0, 0:3], p[0, 0:3], p[1, 0:3], p[1, 0:3]), axis=0).reshape(4, 3)
    # T = np.linalg.solve(np.dot(P_mat.T, P_mat), np.dot(P_mat.T, cons))
    return T
