# fast_slam

Clone this repository into ~/catkin_ws/src to start using it. Then run "cd ~/catkin_ws/" and "catkin build" in the terminal to start using launch files etc.
If wanting to use some of the custom nodes also run: "cd ~/catkin_ws/src/fast_slam/src/fast_slam" and "chmod +x "name of node".py

See how_to.txt for package dependecies and instructions on how to get things to work

# Creating stuff to use in the project

1. Create new .py files inside ~/catkin_ws/src/fast_slam/src/fast_slam

2. As you need to add messages (or code) from other packages, you should include them in the package.xml file. For example, if you want to use the [geometry_msgs](http://wiki.ros.org/geometry_msgs) package, you should add the following line to the package.xml file
    ```
    <depend>geometry_msgs</depend>
    ```

3. As you need to add message (or code) from other packages, you should include them also in the CMakeLists.txt file. This may seem like duplicate work, but it is needed. For example, if you want to use the [geometry_msgs](http://wiki.ros.org/geometry_msgs) package, you should add the following line to the package.xml file
    ```
    find_package(catkin REQUIRED COMPONENTS
        geometry_msgs
    )

    catkin_package(
        CATKIN_DEPENDS 
        geometry_msgs 
    )
    ```

4. If you need to create custom message files, you should add them to the msg folder. For example, if you want to create a custom message called MyMessage, you should create a file called MyMessage.msg inside the msg folder. Then, you should add the following line to the CMakeLists.txt file
    ```
    add_message_files(
        FILES
        MyMessage.msg
    )
    ```

8. Create a Python ROS node program. For that, inside the ~/catkin_ws/src/fast_slam/src/fast_slam folder. 

9. In order for the python node to be executable by ROS, you need to make it executable
    ```
    chmod +x demo_node.py
    ```
    
10. To compile and index the package by the ROS system, run:
    ```
    cd ~/catkin_ws/
    catkin build
    ```
11. Add the following lines to your bashrc file if not already done, in order for the system to know that your code exists and can be executed:
    ```
    nano ~/.bashrc
    
    # Add this line to the .bashrc file and save
    source $HOME/catkin_ws/devel/setup.bash
    ```
12. Source the modified .bashrc file:
    ```
    source ~/.bashrc
    ```
13. Try to run the node by running:
    ```
    rosmaster
    rosrun fast_slam <name of node>.py
    ```

14. Alternatively, use the launch file (and with it, you can avoid having to launch a roscore manually):
    ```
    roslaunch fast_slam <name of launch file>.launch
    ```
