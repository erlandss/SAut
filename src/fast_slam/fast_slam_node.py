#!/usr/bin/env python3

import math
import numpy as np
import random as random
import tkinter as tk
import rospy
from nav_msgs.msg import Odometry
from fiducial_msgs.msg import FiducialTransformArray
import fastSlam1 as fs
from fastSlam1 import Particle, ParticleSet

class FastSlamNode:

    def __init__(self):

        self.Nparticles = 200

        self.detected_aruco_markers = {}

        self.particleSet = ParticleSet(self.Nparticles)
        for i in range(self.Nparticles):
            #Make them random
            self.particleSet.add(Particle(3,5*np.array([random.random(),random.random(),0]).astype(float)))
        
        # Initialize the ROS node
        rospy.init_node('slam_node')
        rospy.loginfo_once('Slam node has started')

        # Initialize the publishers and subscribers
        self.initialize_subscribers()
        
        # Initialize the timer with the corresponding interruption to work at a constant rate
        self.initialize_timer()

    def initialize_subscribers(self):
        """
        Initialize the subscribers to the topics.
        """

        # Subscribe to pose and aruco topics
        self.sub_pose = rospy.Subscriber('/pose', Odometry, self.callback_pose_topic)
        self.sub_aruco = rospy.Subscriber('/fiducial_transforms', FiducialTransformArray, self.callback_aruco_topic)

    def initialize_timer(self):
        """
        Here we create a timer to trigger the callback function at a fixed rate.
        """
        self.timer = rospy.Timer(rospy.Duration(0.6), self.timer_callback)
        self.h_timerActivate = True

    def timer_callback(self, timer):
        """Here you should invoke methods to perform the logic computations of your algorithm.
        Note, the timer object is not used here, but it is passed as an argument to the callback by default.
        This callback is called at a fixed rate as defined in the initialization of the timer.

        At the end of the calculations of your EKF, UKF, Particle Filer, or SLAM algorithm,
        you should publish the results to the corresponding topics.
        """

        # Do something here at a fixed rate
        pass


    def callback_pose_topic(self, msg):
        """
        Callback function for the subscriber of the topic '/aruco_topic'.
        """
        try:
            delta_t = msg.header.stamp.secs + msg.header.stamp.nsecs/1000000000.0 - self.pose_ts
        except:
            delta_t = 0.1
        self.pose_ts = msg.header.stamp.secs + msg.header.stamp.nsecs/1000000000.0
        twist = msg.twist.twist
        u = [twist.linear.x, twist.angular.z]
        for k in range(self.Nparticles):
            fs.predict(u, self.particleSet, delta_t, k)
        rospy.loginfo('Prediction step with linear velocity %f and angular velocity %f', u[0], u[1])
    
    def callback_aruco_topic(self, msg):
        """
        Callback function for the subscriber of the topic '/aruco_topic'.
        """

        pass

def quaternion_to_yaw(quaternion):
    # Extract the yaw angle from the quaternion
    yaw = math.atan2(2 * (quaternion[0] * quaternion[1] + quaternion[2] * quaternion[3]),
                     1 - 2*(quaternion[1]**2 + quaternion[3]**2)) 
    return yaw

class Animater:
    def __init__(self):

        # Initialize some necessary variables here
        self.drawing_scale = 65
        self.drawing_start = (200, 500)
        #Some GUI to see if points are being updated
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=1200, height=800)
        self.canvas.pack()

    def update_display(self):
        self.canvas.delete('all')
        self.draw_arrow(self.pos[0], self.pos[1], self.yaw)
        self.draw_markers()



    def draw_arrow(self, x, y, angle):
        angle = angle + math.pi/2

        marker_length = 15  # Length of the arrow
        
        # Calculate the coordinates of the arrow points
        x = self.drawing_start[0] + self.drawing_scale * x
        y = self.drawing_start[1] - self.drawing_scale * y
        x1 = x - 0.8*marker_length * math.cos(angle)
        y1 = y - 0.8*marker_length * math.sin(angle)
        x2 = x + 0.8*marker_length * math.cos(angle)
        y2 = y + 0.8*marker_length * math.sin(angle)
        
        # Calculate the coordinates of the arrowhead
        x3 = x + 2*marker_length * math.cos(angle + math.pi/2)
        y3 = y + 2*marker_length * math.sin(angle + math.pi/2)
        
        # Draw the arrow on the canvas
        self.canvas.create_polygon(x1, y1, x2, y2, x3, y3, fill="red")

    def draw_markers(self, markers):
        # Draw a circle at each point
        for name, (x, y, c) in markers.items():
            x = self.drawing_start[0] + self.drawing_scale * x
            y = self.drawing_start[1] + self.drawing_scale * y
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='red')
            self.canvas.create_text(x, y-10, text=name)


        
    
def main():

    # Create an instance of the ArucoNode class
    pose_node = FastSlamNode()

    # Start the main loop to update the display
    pose_node.root.mainloop()

    # Spin to keep the script for exiting
    rospy.spin()

if __name__ == '__main__':
    main()
