# coding=utf-8
import json

import math
import numpy as np
import rospy
from nav_msgs.msg import Odometry
import serial

UI = True
if UI:
    import cv2

map_grid_bgr = np.load('data/map.npy')
fr = open('data/path.txt')
parameter = json.load(fr)
fr.close()
pos_x, pos_y, d_x, d_y, direction_x, direction_y = 0, 0, 0, 0, 0, 0

AHEAD = 5
MAP_PIXEL_UNIT = parameter['MAP_PIXEL_UNIT']
ORIGIN_X = parameter['ORIGIN'][1]
ORIGIN_Y = parameter['ORIGIN'][0]
PATH = parameter['PATH']
GOAL_IDX = 0
THRESHOLD_DISTANCE = 1
try:
    ser = serial.Serial('/dev/ttyUSB0', 9600)
except BaseException as e:
    print(e)


def dis(Vec2f_1, Vec2f_2):
    return math.sqrt((Vec2f_1[0] - Vec2f_2[0]) * (Vec2f_1[0] - Vec2f_2[0]) + (Vec2f_1[1] - Vec2f_2[1]) * (Vec2f_1[1] - Vec2f_2[1]))


# current position (Tx,Ty)
# def find_local_goal(Tx, Ty):
#     min_dis = 999.99
#     goal_idx = 0
#     for i in range(len(PATH)):
#         dis_ = dis(PATH[i], (Tx, Ty))
#         if dis_ < min_dis:
#             min_dis = dis_
#             goal_idx = i + AHEAD
#         if goal_idx > len(PATH) - 1:
#             goal_idx = len(PATH) - 1
#     return PATH[goal_idx]
def find_local_goal(Tx, Ty):
    global GOAL_IDX, MAP_PIXEL_UNIT
    if dis(PATH[GOAL_IDX], (Tx, Ty)) * MAP_PIXEL_UNIT > THRESHOLD_DISTANCE + 0.5:
        min_dis = 999.99
        for i in range(len(PATH)):
            dis_ = dis(PATH[i], (Tx, Ty))
            if dis_ < min_dis:
                min_dis = dis_
                GOAL_IDX = i
    while dis(PATH[GOAL_IDX], (Tx, Ty)) * MAP_PIXEL_UNIT < THRESHOLD_DISTANCE and GOAL_IDX < len(PATH) - 1:
        GOAL_IDX += 1
    return PATH[GOAL_IDX]


def quaternion_to_rotation_matrix(x, y, z, w):
    rotation_matrix = np.array(
        [[1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * z * w, 2.0 * x * z + 2.0 * y * w, 0.0],
         [2.0 * x * y + 2.0 * z * w, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * x * w, 0.0],
         [2.0 * x * z - 2.0 * y * w, 2.0 * y * z + 2.0 * x * w, 1.0 - 2.0 * x * x - 2.0 * y * y, 0.0],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return rotation_matrix


def rotate(rotation_matrix, direction_vector):
    rotated_direction_vector = np.dot(rotation_matrix, direction_vector)
    return rotated_direction_vector


# 计算两向量夹角，余弦定理确定旋转角度，外积确定旋转方向
def vector_2_angle(x1, y1, x2, y2):
    A = np.array([x1, y1]).reshape((-1, 1))
    B = np.array([x2, y2]).reshape((-1, 1))
    A_m = np.sqrt(A.T.dot(A))[0, 0]
    B_m = np.sqrt(B.T.dot(B))[0, 0]

    if not A_m or not B_m:
        return 0.0

    A_dot_B = A.T.dot(B)[0, 0]
    cos_angle_A_B = A_dot_B / (A_m * B_m)
    if cos_angle_A_B > 1.0:
        cos_angle_A_B = 1.0
    if cos_angle_A_B < -1.0:
        cos_angle_A_B = -1.0
    radian = math.acos(cos_angle_A_B)

    if x1 * y2 - x2 * y1 < 0:
        radian = -radian

    return radian


def radian_2_angle(radian):
    return radian / math.pi * 180


def callback(data):
    global map_grid_bgr, pos_x, pos_y, d_x, d_y, direction_x, direction_y

    tx = float(data.pose.pose.position.x)
    ty = float(data.pose.pose.position.y)
    tz = float(data.pose.pose.position.z)
    qx = float(data.pose.pose.orientation.x)
    qy = float(data.pose.pose.orientation.y)
    qz = float(data.pose.pose.orientation.z)
    qw = float(data.pose.pose.orientation.w)
    camera_2_world = quaternion_to_rotation_matrix(qx, qy, qz, qw)

    if UI:
        cv2.line(map_grid_bgr, (pos_x, pos_y), (pos_x + d_x, pos_y + d_y), (0, 0, 0), 1)
        cv2.line(map_grid_bgr, (pos_x, pos_y), (pos_x + direction_x, pos_y + direction_y), (0, 0, 0), 1)

    pos_x = int(tx / MAP_PIXEL_UNIT + ORIGIN_X)
    pos_y = int(ty / MAP_PIXEL_UNIT + ORIGIN_Y)
    goal_y, goal_x = find_local_goal(pos_y, pos_x)

    z_vector = np.array([0, 0, 20, 1]).reshape((-1, 1))
    z_vector = rotate(camera_2_world, z_vector)
    d_x = int(z_vector[0, 0])
    d_y = int(z_vector[1, 0])
    direction_x = goal_x - pos_x
    direction_y = goal_y - pos_y

    if UI:
        cv2.line(map_grid_bgr, (pos_x, pos_y), (pos_x + d_x, pos_y + d_y), (0, 255, 0), 1)
        cv2.line(map_grid_bgr, (pos_x, pos_y), (pos_x + direction_x, pos_y + direction_y), (0, 0, 255), 1)

    radian = vector_2_angle(d_x, d_y, direction_x, direction_y)
    angle = radian_2_angle(radian)
    angle = - angle
    # rospy.loginfo(angle)

    angle_output = '{:+.2f}'.format(angle).zfill(7)

    if -10 < angle < 10:
        velocity = 2
    else:
        velocity = 1
    distance = dis((goal_x, goal_y), (pos_x, pos_y)) * MAP_PIXEL_UNIT
    print(goal_x, goal_y)
    print(pos_x, pos_y)
    print(distance)
    if distance <0.2:
        velocity = 0
    elif distance < 2:
       # velocity = int(velocity * distance / 2)
         velocity = 1
   # velocity_output = '{}'.format(velocity).zfill(4)
   # velocity_output = 1	

    out = '1:start\n2:{},{}\n'.format(angle_output, velocity)
    print(out)
    try:
        ser.write(out.encode())
    except BaseException as e:
        print(e)


if __name__ == '__main__':
    print("waiting for vins_estimator/odometry data...")
    rospy.init_node('local_plan', anonymous=True)
    # rospy.Subscriber("/vins_estimator/odometry", Odometry, callback)
    # rospy.Subscriber("/loop_fusion/odometry_rect", Odometry, callback)
    rospy.Subscriber("/init/odometry_map", Odometry, callback)

    while UI and not rospy.is_shutdown():
        cv2.imshow('map', map_grid_bgr.transpose((1, 0, 2))[::-1, ::-1])
        cv2.waitKey(100)

    rospy.spin()
