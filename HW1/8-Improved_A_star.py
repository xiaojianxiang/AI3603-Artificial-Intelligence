import DR20API
import numpy as np

### START CODE HERE ###
# This code block is optional. You can define your utility function and class in this block if necessary.

import matplotlib.pyplot as plt
from math import sqrt
open_list = []
close_list = []
Path = []

def Function_H(current_pos, goal_pos):
    X = abs(current_pos[0] - goal_pos[0])
    Y = abs(current_pos[1] - goal_pos[1])
    Min = X
    Max = Y
    if Min > Y:
        Min = Y
        Max = X
    h = Min*sqrt(2) + Max - Min
    return h

def Draw_Result(current_map, path):
    obstacles_x, obstacles_y = [], []
    for i in range(120):
        for j in range(120):
            if current_map[i][j] == 1:
                obstacles_x.append(i)
                obstacles_y.append(j)

    path_x, path_y = [], []
    for path_node in path:
        path_x.append(path_node[0])
        path_y.append(path_node[1])

    plt.plot(path_x, path_y, "-r")
    plt.plot(current_pos[0], current_pos[1], "xr")
    plt.plot(goal_pos[0], goal_pos[1], "xb")
    plt.plot(obstacles_x, obstacles_y, ".k")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

###  END CODE HERE  ###

def Improved_A_star(current_map, current_pos, goal_pos):
    """
    Given current map of the world, current position of the robot and the position of the goal, 
    plan a path from current position to the goal using improved A* algorithm.

    Arguments:
    current_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    path -- A N*2 array representing the planned path by improved A* algorithm.
    """

    ### START CODE HERE ###

    path = [[current_pos[0], current_pos[1]]]
    point_now = [current_pos[0], current_pos[1]]

    for k in range(3):
        direction_x = [1, 0, -1, 0, 1, 1, -1, -1]
        direction_y = [0, 1, 0, -1, 1, -1, 1, -1]
        
        expand_list = []
        expand_point = point_now
        expand_point_H = 1008610086

        for i in range(8):
            pos_x = point_now[0] + direction_x[i]
            pos_y = point_now[1] + direction_y[i]

            if pos_x <=0 or pos_y <=0:
                continue

            if current_map[pos_x][pos_y] == 1 or [pos_x, pos_y] in close_list:
                continue

            expand_list.append([pos_x, pos_y])
        
        for point in expand_list:
            H = Function_H(point, goal_pos)
            if H < expand_point_H:
                expand_point = point
                expand_point_H = H

        path.append(expand_point)

        close_list.append(point_now)
        point_now = expand_point

    print(path)

    ###  END CODE HERE  ###
    return path

def reach_goal(current_pos, goal_pos):
    """
    Given current position of the robot, 
    check whether the robot has reached the goal.

    Arguments:
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    is_reached -- A bool variable indicating whether the robot has reached the goal, where True indicating reached.
    """

    ### START CODE HERE ###

    is_reached = True
    if abs(current_pos[0] - goal_pos[0]) > 10:
        is_reached = False
    if abs(current_pos[1] - goal_pos[1]) > 10:
        is_reached = False

    ###  END CODE HERE  ###
    return is_reached

if __name__ == '__main__':
    # Define goal position of the exploration, shown as the gray block in the scene.
    goal_pos = [100, 100]
    controller = DR20API.Controller()

    # Initialize the position of the robot and the map of the world.
    current_pos = controller.get_robot_pos()
    current_map = controller.update_map()

    # Plan-Move-Perceive-Update-Replan loop until the robot reaches the goal.
    while not reach_goal(current_pos, goal_pos):
        # Plan a path based on current map from current position of the robot to the goal.
        path = Improved_A_star(current_map, current_pos, goal_pos)
        Path = Path + path[0:-3]
        # Move the robot along the path to a certain distance.
        controller.move_robot(path)
        # Get current position of the robot.
        current_pos = controller.get_robot_pos()
        # Update the map based on the current information of laser scanner and get the updated map.
        current_map = controller.update_map()

    Draw_Result(current_map, Path)
    # Stop the simulation.
    controller.stop_simulation()