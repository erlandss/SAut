#!/usr/bin/env python3

import math
import tkinter as tk
import rospy
from nav_msgs.msg import Odometry

class PoseNode:

    def __init__(self):

        # Initialize some necessary variables here
        self.node_frequency = 30

        self.pos = (200, 500)
        self.yaw = 0
        #Some GUI to see if points are being updated
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=1200, height=800)
        self.canvas.pack()  
   

        # Store the data received from a fake sensor
        self.fake_sensor = 0.0
        
        # Initialize the ROS node
        rospy.init_node('pose_node')
        rospy.loginfo_once('Pose node has started')

        # Initialize the publishers and subscribers
        self.initialize_subscribers()
        
        # Initialize the timer with the corresponding interruption to work at a constant rate
        self.initialize_timer()

    def initialize_subscribers(self):
        """
        Initialize the subscribers to the topics.
        """

        # Subscribe to the topic '/fake_sensor_topic'
        self.sub_fake_sensor_topic = rospy.Subscriber('/pose', Odometry, self.callback_pose_topic)

    def initialize_timer(self):
        """
        Here we create a timer to trigger the callback function at a fixed rate.
        """
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        self.h_timerActivate = True

    def timer_callback(self, timer):
        """Here you should invoke methods to perform the logic computations of your algorithm.
        Note, the timer object is not used here, but it is passed as an argument to the callback by default.
        This callback is called at a fixed rate as defined in the initialization of the timer.

        At the end of the calculations of your EKF, UKF, Particle Filer, or SLAM algorithm,
        you should publish the results to the corresponding topics.
        """

        # Do something here at a fixed rate
        self.draw_arrow(self.pos[0], self.pos[1], self.yaw)


    def callback_pose_topic(self, msg):
        """
        Callback function for the subscriber of the topic '/aruco_topic'. This function is called
        whenever a message is received by the subscriber.
        """
        point = msg.pose.pose.position #x, y, z
        quaternion = msg.pose.pose.orientation #x, y, z, w
        self.pos = (point.x, point.y)
        self.yaw = quaternion_to_yaw([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        rospy.loginfo('x: %f, y: %f, z: %f, w: %f, yaw: %f', quaternion.x, quaternion.y, quaternion.z, quaternion.w, self.yaw)


    def draw_arrow(self, x, y, angle):
        self.canvas.delete('all')
        x = 200 + 75*x
        y = 500 - 75*y
        angle = angle + math.pi/2

        marker_length = 15  # Length of the arrow
        
         # Calculate the coordinates of the arrow points
        x1 = x - 0.8*marker_length * math.cos(angle)
        y1 = y - 0.8*marker_length * math.sin(angle)
        x2 = x + 0.8*marker_length * math.cos(angle)
        y2 = y + 0.8*marker_length * math.sin(angle)
        
        # Calculate the coordinates of the arrowhead
        x3 = x + 2*marker_length * math.cos(angle + math.pi/2)
        y3 = y + 2*marker_length * math.sin(angle + math.pi/2)
        
        # Draw the arrow on the canvas
        self.canvas.create_polygon(x1, y1, x2, y2, x3, y3, fill="red")


def quaternion_to_yaw(quaternion):
    # Extract the yaw angle from the quaternion
    yaw = math.atan2(2 * (quaternion[0] * quaternion[1] + quaternion[2] * quaternion[3]),
                     1 - 2*(quaternion[1]**2 + quaternion[2]**2)) 
    return yaw


        
    
def main():

    # Create an instance of the ArucoNode class
    pose_node = PoseNode()

    # Start the main loop to update the display
    pose_node.root.mainloop()

    # Spin to keep the script for exiting
    rospy.spin()

if __name__ == '__main__':
    main()
