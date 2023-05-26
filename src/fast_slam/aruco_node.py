#!/usr/bin/env python3

import tkinter as tk
import rospy
from fiducial_msgs.msg import FiducialTransformArray

class ArucoNode:

    def __init__(self):

        # Initialize some necessary variables here
        self.node_frequency = 30
        self.sub_aruco_topic = None
        self.detected_aruco_markers = {}
        #Some GUI to see if points are being updated
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=1200, height=800)
        self.canvas.pack()       

        # Store the data received from a fake sensor
        self.fake_sensor = 0.0
        
        # Initialize the ROS node
        rospy.init_node('aruco_node')
        rospy.loginfo_once('Aruco node has started')

        # Initialize the publishers and subscribers
        self.initialize_subscribers()
        
        # Initialize the timer with the corresponding interruption to work at a constant rate
        self.initialize_timer()

    def initialize_subscribers(self):
        """
        Initialize the subscribers to the topics.
        """

        # Subscribe to the topic '/fake_sensor_topic'
        self.sub_fake_sensor_topic = rospy.Subscriber('/fiducial_transforms', FiducialTransformArray, self.callback_aruco_topic)

    def initialize_timer(self):
        """
        Here we create a timer to trigger the callback function at a fixed rate.
        """
        self.timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)
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
        rospy.loginfo('Number of aruco markers detected: %i', len(self.detected_aruco_markers))
        self.update_display()


    def callback_aruco_topic(self, msg):
        """
        Callback function for the subscriber of the topic '/aruco_topic'. This function is called
        whenever a message is received by the subscriber.
        """

        for t in msg.transforms:
            marker_id = t.fiducial_id
            rospy.loginfo('Received detection of marker: %s', t.fiducial_id)
            x = t.transform.translation.z
            y = -t.transform.translation.y
            if marker_id in self.detected_aruco_markers:
                self.detected_aruco_markers[marker_id] = (x,y)
            else:
                self.detected_aruco_markers[marker_id] = (x,y)
    
    def update_display(self):
        # Clear the canvas
        self.canvas.delete('all')
        # Draw a circle at each point
        for name, (x, y) in self.detected_aruco_markers.items():
            x = x*100+400
            y = y*100+400
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='red')
            self.canvas.create_text(x, y-10, text=name)
        

def main():

    # Create an instance of the ArucoNode class
    aruco_node = ArucoNode()
    # Start the main loop to update the display
    aruco_node.root.mainloop()

    # Spin to keep the script for exiting
    rospy.spin()

if __name__ == '__main__':
    main()
