#!/usr/bin/env python3
# Sends one robot back and forth between two points on /<robot>/term_goal.
# Usage: swap_goals.py <robot> <x1> <y1> <x2> <y2>   (starts with x1 y1)
import math
import sys

import rclpy
from dynus_interfaces.msg import State
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

TOLERANCE = 1.0


class SwapGoals(Node):
    def __init__(self, robot, goals):
        super().__init__('swap_goals', namespace=robot)
        self.goals = goals
        self.index = 0
        self.pub = self.create_publisher(PoseStamped, 'term_goal', 10)
        self.create_subscription(State, 'state', self.on_state, 10)
        self.create_timer(1.0, self.publish_goal)

    def on_state(self, msg):
        x, y = self.goals[self.index]
        if math.hypot(msg.pos.x - x, msg.pos.y - y) < TOLERANCE:
            self.index = 1 - self.index
            self.get_logger().info(f'reached ({x}, {y}), next {self.goals[self.index]}')
            self.publish_goal()

    def publish_goal(self):
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x, goal.pose.position.y = self.goals[self.index]
        goal.pose.position.z = 0.2
        goal.pose.orientation.w = 1.0
        self.pub.publish(goal)


def main():
    robot = sys.argv[1]
    x1, y1, x2, y2 = map(float, sys.argv[2:6])
    rclpy.init()
    rclpy.spin(SwapGoals(robot, [(x1, y1), (x2, y2)]))


if __name__ == '__main__':
    main()
