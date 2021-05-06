# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import math
import tensorflow as tf


def forward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [x_c, y_c, w, h, theta]
    :return: format [x1, y1, x2, y2, x3, y3, x4, y4]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], rect[5]])
    else:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])

    return np.array(boxes, dtype=np.float32)


def backward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)]
    :param with_label: default True
    :return: format [x_c, y_c, w, h, theta, (label)]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = np.int0(rect[:-1])
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]

            if theta == 0:
                w, h = h, w
                theta -= 90

            boxes.append([x, y, w, h, theta, rect[-1]])

    else:
        for rect in coordinate:
            box = np.int0(rect)
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]

            if theta == 0:
                w, h = h, w
                theta -= 90

            boxes.append([x, y, w, h, theta])

    return np.array(boxes, dtype=np.float32)


def get_horizen_minAreaRectangle(boxes, with_label=True):

    if with_label:
        boxes = tf.reshape(boxes, [-1, 9])

        boxes_shape = tf.shape(boxes)
        x_list = tf.strided_slice(boxes, begin=[0, 0], end=[boxes_shape[0], boxes_shape[1] - 1],
                                  strides=[1, 2])
        y_list = tf.strided_slice(boxes, begin=[0, 1], end=[boxes_shape[0], boxes_shape[1] - 1],
                                  strides=[1, 2])

        label = tf.unstack(boxes, axis=1)[-1]

        y_max = tf.reduce_max(y_list, axis=1)
        y_min = tf.reduce_min(y_list, axis=1)
        x_max = tf.reduce_max(x_list, axis=1)
        x_min = tf.reduce_min(x_list, axis=1)
        return tf.transpose(tf.stack([x_min, y_min, x_max, y_max, label], axis=0))
    else:
        boxes = tf.reshape(boxes, [-1, 8])

        boxes_shape = tf.shape(boxes)
        x_list = tf.strided_slice(boxes, begin=[0, 0], end=[boxes_shape[0], boxes_shape[1]],
                                  strides=[1, 2])
        y_list = tf.strided_slice(boxes, begin=[0, 1], end=[boxes_shape[0], boxes_shape[1]],
                                  strides=[1, 2])

        y_max = tf.reduce_max(y_list, axis=1)
        y_min = tf.reduce_min(y_list, axis=1)
        x_max = tf.reduce_max(x_list, axis=1)
        x_min = tf.reduce_min(x_list, axis=1)

    return tf.transpose(tf.stack([x_min, y_min, x_max, y_max], axis=0))


def coordinate_present_convert(coords, mode=1, shift=True):
    """
    :param coords: shape [-1, 5]
    :param mode: -1 convert coords range to [-90, 90), 1 convert coords range to [-90, 0)
    :param shift: [-90, 90) --> [-180, 0)
    :return: shape [-1, 5]
    """
    # angle range from [-90, 0) to [-180, 0)
    if mode == -1:
        w, h = coords[:, 2], coords[:, 3]

        remain_mask = np.greater(w, h)
        convert_mask = np.logical_not(remain_mask).astype(np.int32)
        remain_mask = remain_mask.astype(np.int32)

        remain_coords = coords * np.reshape(remain_mask, [-1, 1])

        coords[:, [2, 3]] = coords[:, [3, 2]]
        coords[:, 4] += 90

        convert_coords = coords * np.reshape(convert_mask, [-1, 1])

        coords_new = remain_coords + convert_coords

        if shift:
            coords_new[:, 4] -= 90

    # angle range from [-180, 0) to [-90, 0)
    elif mode == 1:
        if shift:
            coords[:, 4] += 90

        # theta = coords[:, 4]
        # remain_mask = np.logical_and(np.greater_equal(theta, -90), np.less(theta, 0))
        # convert_mask = np.logical_not(remain_mask)
        #
        # remain_coords = coords * np.reshape(remain_mask, [-1, 1])
        #
        # coords[:, [2, 3]] = coords[:, [3, 2]]
        # coords[:, 4] -= 90
        #
        # convert_coords = coords * np.reshape(convert_mask, [-1, 1])
        #
        # coords_new = remain_coords + convert_coords

        xlt, ylt = -1 * coords[:, 2] / 2.0, coords[:, 3] / 2.0
        xld, yld = -1 * coords[:, 2] / 2.0, -1 * coords[:, 3] / 2.0
        xrd, yrd = coords[:, 2] / 2.0, -1 * coords[:, 3] / 2.0
        xrt, yrt = coords[:, 2] / 2.0, coords[:, 3] / 2.0

        theta = -coords[:, -1] / 180 * np.pi

        xlt_ = np.cos(theta) * xlt + np.sin(theta) * ylt + coords[:, 0]
        ylt_ = -np.sin(theta) * xlt + np.cos(theta) * ylt + coords[:, 1]

        xrt_ = np.cos(theta) * xrt + np.sin(theta) * yrt + coords[:, 0]
        yrt_ = -np.sin(theta) * xrt + np.cos(theta) * yrt + coords[:, 1]

        xld_ = np.cos(theta) * xld + np.sin(theta) * yld + coords[:, 0]
        yld_ = -np.sin(theta) * xld + np.cos(theta) * yld + coords[:, 1]

        xrd_ = np.cos(theta) * xrd + np.sin(theta) * yrd + coords[:, 0]
        yrd_ = -np.sin(theta) * xrd + np.cos(theta) * yrd + coords[:, 1]

        coords_new = np.transpose(np.stack([xlt_, ylt_, xrt_, yrt_, xrd_, yrd_, xld_, yld_], axis=0))

        coords_new = backward_convert(coords_new, False)

    else:
        raise Exception('mode error!')

    return np.array(coords_new, dtype=np.float32)


def coordinate5_2_8_tf(coords):
    coords = coordinate90_2_180_tf(coords)

    x, y, h, w, theta = tf.unstack(coords, axis=1)

    xlt, ylt = -1 * h / 2.0, w / 2.0
    xld, yld = -1 * h / 2.0, -1 * w / 2.0
    xrd, yrd = h / 2.0, -1 * w / 2.0
    xrt, yrt = h / 2.0, w / 2.0

    xlt_ = tf.cos(theta) * xlt + tf.sin(theta) * ylt + x
    ylt_ = -tf.sin(theta) * xlt + tf.cos(theta) * ylt + y

    xrt_ = tf.cos(theta) * xrt + tf.sin(theta) * yrt + x
    yrt_ = -tf.sin(theta) * xrt + tf.cos(theta) * yrt + y

    xld_ = tf.cos(theta) * xld + tf.sin(theta) * yld + x
    yld_ = -tf.sin(theta) * xld + tf.cos(theta) * yld + y

    xrd_ = tf.cos(theta) * xrd + tf.sin(theta) * yrd + x
    yrd_ = -tf.sin(theta) * xrd + tf.cos(theta) * yrd + y

    convert_box = tf.transpose(tf.stack([xlt_, ylt_, xrt_, yrt_, xrd_, yrd_, xld_, yld_], axis=0))
    return convert_box


def coordinate90_2_180(coords):

    # angle range from [-90, 0) to [-180, 0)
    w, h = coords[:, 2], coords[:, 3]

    remain_mask = np.greater(w, h)
    convert_mask = np.logical_not(remain_mask).astype(np.int32)
    remain_mask = remain_mask.astype(np.int32)

    remain_coords = coords * np.reshape(remain_mask, [-1, 1])

    coords[:, [2, 3]] = coords[:, [3, 2]]
    coords[:, 4] += 90

    convert_coords = coords * np.reshape(convert_mask, [-1, 1])

    coords_new = remain_coords + convert_coords

    coords_new[:, 4] *= (-np.pi / 180)

    return coords_new


def coordinate90_2_180_tf(coords, is_radian=True, change_range=False):

    # angle range from [-90, 0) to [-180, 0)
    x, y, w, h, theta = tf.unstack(coords, axis=1)

    remain_mask = tf.greater(w, h)
    convert_mask = tf.cast(tf.logical_not(remain_mask), tf.int32)
    remain_mask = tf.cast(remain_mask, tf.int32)

    remain_coords = coords * tf.cast(tf.reshape(remain_mask, [-1, 1]), tf.float32)

    h_ = tf.cast(w, tf.float32)
    w_ = tf.cast(h, tf.float32)
    theta += 90.

    coords_ = tf.transpose(tf.stack([x, y, w_, h_, theta]))

    convert_coords = coords_ * tf.cast(tf.reshape(convert_mask, [-1, 1]), tf.float32)

    coords_new = remain_coords + convert_coords
    x_new, y_new, w_new, h_new, theta_new = tf.unstack(coords_new, axis=1)

    if change_range:
        theta_new -= 90

    if is_radian:
        theta_new *= (-np.pi / 180)

    coords_new_ = tf.transpose(tf.stack([x_new, y_new, w_new, h_new, theta_new]))

    return coords_new_


def coords_regular(coords):
    # [-180, -90) -> [-90, 90)
    theta = coords[:, 4]
    convert_mask = np.logical_and(np.greater_equal(theta, -180), np.less(theta, -90))
    remain_mask = np.logical_not(convert_mask)
    remain_coords = coords * np.reshape(remain_mask, [-1, 1])

    coords[:, [2, 3]] = coords[:, [3, 2]]
    coords[:, 4] -= 90

    convert_coords = coords * np.reshape(convert_mask, [-1, 1])
    coords_new = remain_coords + convert_coords
    return coords_new


if __name__ == '__main__':
    # coord = np.array([[150, 150, 50, 100, -90, 1],
    #                   [150, 150, 100, 50, -90, 1],
    #                   [150, 150, 50, 100, -45, 1],
    #                   [150, 150, 100, 50, -45, 1]])
    #
    # coord1 = np.array([[150, 150, 100, 50, 0],
    #                   [150, 150, 100, 50, -90],
    #                   [150, 150, 100, 50, 45],
    #                   [150, 150, 100, 50, -45]])
    #
    # coord2 = forward_convert(coord)
    # coord3 = forward_convert(coord1, mode=-1)
    # print(coord2)
    # print(coordinate_present_convert(coord, mode=-1))

    coord3 = np.array([[150, 150, 150, 20, -95],
                      [150, 150, 20, 150, -5]], dtype=np.float32)
    print(forward_convert(coord3, False))

    # coord4 = np.array([[0, 0, 0, 180, 20, 180, 20, 0],
    #                    [10, 0, 0, 20, 180, 20, 180, 0]], dtype=np.float32)
    # print(backward_convert(coord4, False))
    # coord5 = np.array([[-3.0636845, 1.479856, 51.584435, 33.250473, -17.427048]])
    # print(coordinate_present_convert(coord5, 1))

    # coord4 = coordinate5_2_8_tf(tf.convert_to_tensor(coord3))
    # coord5 = coordinate_present_convert(coordinate_present_convert(coord3, mode=-1), mode=1)
    # print(coord5)
    #
    # with tf.Session() as sess:
    #     coord4_ = sess.run([coord4])
    #     print(coord4_)
        # print(backward_convert(coord4_, False))

