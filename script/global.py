# coding=utf-8
import json
import sys

import cv2
import numpy as np

MAP_PIXEL_UNIT = 0.1
UI = True


class AStar(object):
    """
    创建一个A*算法类
    """

    def __init__(self, start, goal, map_grid, max_iter):
        """
        初始化
        """
        self.start = np.array(start)  # 起点坐标
        self.goal = np.array(goal)  # 终点坐标
        self.open = np.array([[], [], [], [], [], []])  # 先创建一个空的open表, 记录坐标，方向，g值，f值
        self.closed = np.array([[], [], [], [], [], []])  # 先创建一个空的closed表
        self.best_path_array = np.array([[], []])  # 回溯路径表
        self.map_grid = map_grid
        self.map_width = map_grid.shape[1]
        self.map_height = map_grid.shape[0]
        self.max_iter = max_iter

    def h_value_tem(self, son_p):
        """
        计算拓展节点和终点的h值
        :param son_p:子搜索节点坐标
        :return:
        """
        h = (son_p[0] - self.goal[0]) ** 2 + (son_p[1] - self.goal[1]) ** 2
        h = np.sqrt(h)  # 计算h
        return h

    def g_accumulation(self, son_point, father_point):
        """
        累计的g值
        :return:
        """
        g1 = father_point[0] - son_point[0]
        g2 = father_point[1] - son_point[1]
        g = g1 ** 2 + g2 ** 2
        g = np.sqrt(g) + father_point[4]  # 加上累计的g值
        return g

    def f_value_tem(self, son_p, father_p):
        """
        求出的是临时g值和h值加上累计g值得到全局f值
        :param father_p: 父节点坐标
        :param son_p: 子节点坐标
        :return:f
        """
        f = self.g_accumulation(son_p, father_p) + self.h_value_tem(son_p)
        return f

    def child_point(self, x):
        """
        拓展的子节点坐标
        :param x: 父节点坐标
        :return: 子节点存入open表，返回值是每一次拓展出的子节点数目，用于撞墙判断
        当搜索的节点撞墙后，如果不加处理，会陷入死循环
        """
        # 开始遍历周围8个节点
        for j in range(-1, 2, 1):
            for q in range(-1, 2, 1):

                if j == 0 and q == 0:  # 搜索到父节点去掉
                    continue
                m = [x[0] + j, x[1] + q]
                # print(m)
                if m[0] < 0 or m[0] > self.map_height or m[1] < 0 or m[1] > self.map_width:  # 搜索点出了边界去掉
                    continue

                if self.map_grid[int(m[0]), int(m[1])] == 0:  # 搜索到障碍物去掉
                    continue

                record_g = self.g_accumulation(m, x)
                record_f = self.f_value_tem(m, x)  # 计算每一个节点的f值

                x_direction, y_direction = self.direction(x, m)  # 每产生一个子节点，记录一次方向

                para = [m[0], m[1], x_direction, y_direction, record_g, record_f]  # 将参数汇总一下
                # print(para)

                # 在open表中，则去掉搜索点，但是需要更新方向指针和self.g值
                # 而且只需要计算并更新self.g即可，此时建立一个比较g值的函数
                a, index = self.judge_location(m, self.open)
                if a == 1:
                    # 说明open中已经存在这个点

                    if record_f <= self.open[5][index]:
                        self.open[5][index] = record_f
                        self.open[4][index] = record_g
                        self.open[3][index] = y_direction
                        self.open[2][index] = x_direction

                    continue

                # 在closed表中,则去掉搜索点
                b, index2 = self.judge_location(m, self.closed)
                if b == 1:

                    if record_f <= self.closed[5][index2]:
                        self.closed[5][index2] = record_f
                        self.closed[4][index2] = record_g
                        self.closed[3][index2] = y_direction
                        self.closed[2][index2] = x_direction
                        self.closed = np.delete(self.closed, index2, axis=1)
                        self.open = np.c_[self.open, para]
                    continue

                self.open = np.c_[self.open, para]  # 参数添加到open中
                # print(self.open)

    def judge_location(self, m, list_co):
        """
        判断拓展点是否在open表或者closed表中
        :return:返回判断是否存在，和如果存在，那么存在的位置索引
        """
        jud = 0
        index = 0
        for i in range(list_co.shape[1]):

            if m[0] == list_co[0, i] and m[1] == list_co[1, i]:

                jud = jud + 1

                index = i
                break
            else:
                jud = jud
        # if a != 0:
        #     continue
        return jud, index

    def direction(self, father_point, son_point):
        """
        建立每一个节点的方向，便于在closed表中选出最佳路径
        非常重要的一步，不然画出的图像参考1.1版本
        x记录子节点和父节点的x轴变化
        y记录子节点和父节点的y轴变化
        如（0，1）表示子节点在父节点的方向上变化0和1
        :return:
        """
        x = son_point[0] - father_point[0]
        y = son_point[1] - father_point[1]
        return x, y

    def path_backtrace(self):
        """
        回溯closed表中的最短路径
        :return:
        """
        best_path = self.goal  # 回溯路径的初始化
        self.best_path_array = np.array([[self.goal[0]], [self.goal[1]]])
        j = 0
        while j <= self.closed.shape[1]:
            for i in range(self.closed.shape[1]):
                if best_path[0] == self.closed[0][i] and best_path[1] == self.closed[1][i]:
                    x = self.closed[0][i] - self.closed[2][i]
                    y = self.closed[1][i] - self.closed[3][i]
                    best_path = [x, y]
                    self.best_path_array = np.c_[self.best_path_array, best_path]
                    break  # 如果已经找到，退出本轮循环，减少耗时
                else:
                    continue
            j = j + 1
        path = []

        path.append((int(self.start[0]), int(self.start[1])))
        for i in range(len(self.best_path_array[0])):
            if abs(self.best_path_array[0][-i - 1] - self.start[0]) + abs(
                    self.best_path_array[1][-i - 1] - self.start[1]) < 0.1:
                continue
            path.append((int(self.best_path_array[0][-i - 1]), int(self.best_path_array[1][-i - 1])))

        return path

    def finding_path(self):
        """
        main函数
        :return:
        """
        best = self.start  # 起点放入当前点，作为父节点
        h0 = self.h_value_tem(best)
        init_open = [best[0], best[1], 0, 0, 0, h0]  # 将方向初始化为（0，0），g_init=0,f值初始化h0
        self.open = np.column_stack((self.open, init_open))  # 起点放入open,open初始化

        ite = 1  # 设置迭代次数小于200，防止程序出错无限循环
        while ite <= self.max_iter:

            # open列表为空，退出
            if self.open.shape[1] == 0:
                print('没有搜索到路径！')
                return

            self.open = self.open.T[np.lexsort(self.open)].T  # open表中最后一行排序(联合排序）

            # 选取open表中最小f值的节点作为best，放入closed表

            best = self.open[:, 0]
            # print('检验第%s次当前点坐标*******************' % ite)
            # print(best)
            self.closed = np.c_[self.closed, best]

            if best[0] == self.goal[0] and best[1] == self.goal[1]:  # 如果best是目标点，退出
                print('搜索成功！')
                return

            self.child_point(best)  # 生成子节点并判断数目
            # print(self.open)
            self.open = np.delete(self.open, 0, axis=1)  # 删除open中最优点

            # print(self.open)

            ite = ite + 1


def generate_map_grid_from_vio_loop(pose_graph_filename):
    fr = open(pose_graph_filename)
    xv = []
    yv = []
    for r in fr.readlines():
        r = r.strip().split(",")
        x = float(r[1]) / MAP_PIXEL_UNIT
        y = float(r[2]) / MAP_PIXEL_UNIT
        xv.append(int(x))
        yv.append(int(y))
    xmin = min(xv)
    xmax = max(xv)
    ymin = min(yv)
    ymax = max(yv)

    MAP_WIDTH = xmax - xmin + 10
    MAP_HEIGHT = ymax - ymin + 10
    print('栅格宽度：{}'.format(MAP_WIDTH))
    print('栅格高度：{}'.format(MAP_HEIGHT))

    # xmin + ORIGIN_X = 5
    ORIGIN_X = -xmin + 5
    # ymin + ORIGIN_Y = 5
    ORIGIN_Y = -ymin + 5
    print (xmin, ymin)
    print (ORIGIN_X,ORIGIN_Y)
    map_grid_bgr = np.zeros((MAP_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)
    #
    cv2.line(map_grid_bgr, (0 + ORIGIN_X, 0 + ORIGIN_Y), (50 + ORIGIN_X, 0 + ORIGIN_Y), (0, 0, 255), 1)
    cv2.line(map_grid_bgr, (0 + ORIGIN_X, 0 + ORIGIN_Y), (0 + ORIGIN_X, 30 + ORIGIN_Y), (0, 255, 0), 1)
 #   cv2.setMouseCallback('map',draw_circle)
    cv2.putText(map_grid_bgr,"x",(50 + ORIGIN_X, 0 + ORIGIN_Y),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0,255),1)
    cv2.putText(map_grid_bgr, "y", (0 + ORIGIN_X, 30 + ORIGIN_Y), cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255,0),1)
    if UI:
        print('请按S确定目标点坐标!')
    goal_x, goal_y, key = 0, 0, -1
    for i, (x, y) in enumerate(zip(xv, yv)):
        try:
            x0 = x + ORIGIN_X
            y0 = y + ORIGIN_Y
            x1 = xv[i + 1] + ORIGIN_X
            y1 = yv[i + 1] + ORIGIN_Y
            # print(x1, y1)
            if abs(x0 - x1) * MAP_PIXEL_UNIT < 10 and abs(y0 - y1) * MAP_PIXEL_UNIT < 10:
                cv2.line(map_grid_bgr, (x0, y0), (x1, y1), (255, 0, 0), 1)

            if UI and not goal_x and not goal_y:# 还没有确认终点
                cv2.imshow('map', map_grid_bgr.transpose((1, 0, 2))[::-1, ::-1])
                cv2.resizeWindow("map", 640, 480)
                key = cv2.waitKey(10)
        # 按S键

            if key == ord('s'):  # 确认终点
                print("+" * 100)
                print('目标点坐标：{},{}'.format(xv[i + 1], yv[i + 1]))
                goal_x, goal_y = xv[i + 1], yv[i + 1]
                print("+" * 100)
                key = -1
        except BaseException as e:
            pass

    if UI and not goal_x and not goal_y: # 还没有确认终点
        print("+" * 100)
        print('未确定目标点坐标，请再次运行按S确定目标点坐标!')
        print("+" * 100)

    return map_grid_bgr, ORIGIN_X, ORIGIN_Y, MAP_WIDTH, MAP_HEIGHT, xv[0], yv[0], goal_x, goal_y


if __name__ == '__main__':
    if len(sys.argv) == 1:
        UI = True
   #     cv2.namedWindow('result', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)

    map_grid_bgr, ORIGIN_X, ORIGIN_Y, MAP_WIDTH, MAP_HEIGHT, start_x, start_y, goal_x, goal_y = \
        generate_map_grid_from_vio_loop('../source/vio_loop.csv')

    start = (ORIGIN_X + start_x, ORIGIN_Y + start_y)

    print("start_x {} start_y {}".format(start_x,start_y))
    print("start {} ".format(start))
    goal = (ORIGIN_X + goal_x, ORIGIN_Y + goal_y)
    if len(sys.argv) == 3:
        goal = (ORIGIN_X + int(sys.argv[1]), ORIGIN_Y + int(sys.argv[2]))


    cv2.line(map_grid_bgr, (0 + ORIGIN_X, -40 + ORIGIN_Y), (0 + ORIGIN_X, 300 + ORIGIN_Y), (0, 0, 0), 1)
    a1 = AStar((start[1], start[0]), (goal[1], goal[0]), map_grid_bgr[:, :, 0], max_iter=20000)
    a1.finding_path()
    PATH = a1.path_backtrace()
    for y, x in PATH:  # 选中的路线 设置为白色线
        map_grid_bgr[y, x] = (255, 255, 255)

    np.save('data/map.npy', map_grid_bgr)
    fw = open("data/path.txt", "w")
    json.dump({'MAP_PIXEL_UNIT': MAP_PIXEL_UNIT, 'ORIGIN': [ORIGIN_Y, ORIGIN_X], 'PATH': PATH}, fw)
    fw.close()

    if UI:
        cv2.imshow('map', map_grid_bgr.transpose((1, 0, 2))[::-1, ::-1])
        cv2.resizeWindow("map", 640, 480)
        cv2.waitKey()
    cv2.destroyAllWindows()
