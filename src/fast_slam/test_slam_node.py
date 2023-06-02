#!/usr/bin/env python3

import math
import tkinter as tk
import rospy
from nav_msgs.msg import Odometry
from fiducial_msgs.msg import FiducialTransformArray

class PoseNode:

    def __init__(self):

        # Initialize some necessary variables here
        self.node_frequency = 30
        self.drawing_scale = 65
        self.drawing_start = (200, 500)
        self.pos = (0,0)
        self.yaw = 0
        self.detected_aruco_markers = {}
        #Some GUI to see if points are being updated
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=1200, height=800)
        self.canvas.pack()
        
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
        self.timer = rospy.Timer(rospy.Duration(0.8), self.timer_callback)
        self.h_timerActivate = True

    def timer_callback(self, timer):
        """Here you should invoke methods to perform the logic computations of your algorithm.
        Note, the timer object is not used here, but it is passed as an argument to the callback by default.
        This callback is called at a fixed rate as defined in the initialization of the timer.

        At the end of the calculations of your EKF, UKF, Particle Filer, or SLAM algorithm,
        you should publish the results to the corresponding topics.
        """

        # Do something here at a fixed rate
        self.update_display()


    def callback_pose_topic(self, msg):
        """
        Callback function for the subscriber of the topic '/aruco_topic'.
        """
        point = msg.pose.pose.position #x, y, z
        quaternion = msg.pose.pose.orientation #x, y, z, w
        self.pos = (point.x, point.y)
        self.yaw = quaternion_to_yaw(quaternion)
        twist = msg.twist.twist
        u = [twist.linear, twist.angular]
        rospy.loginfo('x: %f, y: %f, z: %f, w: %f, yaw: %f', quaternion.x, quaternion.y, quaternion.z, quaternion.w, self.yaw)
        #rospy.loginfo('x: %f, y: %f, z: %f, ax: %f, ay: %f, az: %f', u[0].x, u[0].y, u[0].z, u[1].x, u[1].y, u[1].z)
        #self.pose_ts = msg.header.stamp.secs + msg.header.stamp.nsecs/1000000000.0
        #rospy.loginfo(self.pose_ts)
    
    def callback_aruco_topic(self, msg):
        """
        Callback function for the subscriber of the topic '/aruco_topic'.
        """

        angle = -self.yaw
        for t in msg.transforms:
            marker_id = t.fiducial_id
            x = t.transform.translation.z
            y = -t.transform.translation.y
            rospy.loginfo('Received detection of marker %s at position (%f, %f) relative to robot at robot angle %f',
                           marker_id, x, y, angle)
            x = self.pos[0] + math.cos(angle) * x - math.sin(angle) * y
            y = self.pos[1] + math.sin(angle) * x + math.cos(angle) * y
            if marker_id not in self.detected_aruco_markers:
                self.detected_aruco_markers[marker_id] = (x,y)

    def update_display(self):
        self.canvas.delete('all')
        self.draw_arrow(self.pos[0], self.pos[1], self.yaw)
        self.draw_markers()



    def draw_arrow(self, x, y, angle):
        angle = -(angle+math.pi/2)

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

    def draw_markers(self):
        # Draw a circle at each point
        for name, (x, y) in self.detected_aruco_markers.items():
            x = self.drawing_start[0] + self.drawing_scale * x
            y = self.drawing_start[1] - self.drawing_scale * y
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='red')
            self.canvas.create_text(x, y-10, text=name)


def quaternion_to_yaw(quaternion):
    # Extract the yaw angle from the quaternion
    qw = quaternion.w
    qx = quaternion.x
    qy = quaternion.y
    qz = quaternion.z
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
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
