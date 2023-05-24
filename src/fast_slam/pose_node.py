#!/usr/bin/env python3

import math
import tkinter as tk
import rospy
from nav_msgs.msg import Odometry

class PoseNode:

    def __init__(self):

        # Initialize some necessary variables here
        self.node_frequency = 30

        self.pos = (400, 400)
        self.yaw = 0
        #Some GUI to see if points are being updated
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=800, height=800)
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
        rospy.loginfo('Timer callback called at: %s', rospy.get_time())
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
        rospy.loginfo('Yaw is : %f', self.yaw)


    def draw_arrow(self, x, y, angle_degrees):
        self.canvas.delete('all')
        x = 400 + 100*x
        y = 400 - 100*y

        arrow_length = 50  # Length of the arrow
        arrow_width = 10  # Width of the arrow
        arrow_head_length = 20  # Length of the arrow head
        
        # Convert the angle from degrees to radians
        angle_radians = math.radians(angle_degrees)
        
        # Calculate the endpoint of the arrow
        end_x = x + arrow_length * math.cos(angle_radians)
        end_y = y - arrow_length * math.sin(angle_radians)
        
        # Calculate the coordinates of the arrowhead
        arrowhead_1_x = end_x + arrow_head_length * math.cos(angle_radians + math.pi - arrow_width/2)
        arrowhead_1_y = end_y - arrow_head_length * math.sin(angle_radians + math.pi - arrow_width/2)
        arrowhead_2_x = end_x + arrow_head_length * math.cos(angle_radians + math.pi + arrow_width/2)
        arrowhead_2_y = end_y - arrow_head_length * math.sin(angle_radians + math.pi + arrow_width/2)
        
        # Draw the arrow on the canvas
        self.canvas.create_line(x, y, end_x, end_y, width=2, arrow=tk.LAST)
        self.canvas.create_polygon(end_x, end_y, arrowhead_1_x, arrowhead_1_y, arrowhead_2_x, arrowhead_2_y, fill='black')


def quaternion_to_yaw(quaternion):
    # Extract the yaw angle from the quaternion
    yaw = math.atan2(2 * (quaternion[1] * quaternion[2] + quaternion[0] * quaternion[3]),
                    quaternion[0]**2 - quaternion[1]**2 - quaternion[2]**2 + quaternion[3]**2)
    
    # Convert the yaw angle from radians to degrees
    yaw_degrees = math.degrees(yaw)
    
    # Adjust the yaw angle to be within the range of -180 to 180 degrees
    if yaw_degrees > 180:
        yaw_degrees -= 360
    elif yaw_degrees < -180:
        yaw_degrees += 360
    
    return yaw_degrees


        
    
def main():

    # Create an instance of the ArucoNode class
    pose_node = PoseNode()

    # Start the main loop to update the display
    pose_node.root.mainloop()

    # Spin to keep the script for exiting
    rospy.spin()

if __name__ == '__main__':
    main()
