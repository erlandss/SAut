# SAut

- Should install these ones:
sudo apt-get install python3-rosdep
rosdep init
sudo apt-get install python3-catkin-tools


- Add lines to bashrc:
nano ~/.bashrc
- Press Ctrl+w+v to reach bottom and paste the following lines:
source /opt/ros/noetic/setup.bash
alias s="$HOME/catkin_ws/src/devel/setup.bash"
source $HOME/catkin_ws/devel/setup.bash
- Save and exit and source the modified .bashrc file:
source ~/.bashrc


- After installing new packages:
cd ~/catkin_ws/
catkin build


- To use camera on VMware, set USB compatibility to 3.1 in the USB Controller in Devices


- Publish topics from camera:
dmesg
- Use above to find serial number of camera and run:
roslaunch freenect_launch freenect.launch device_id:=<serial number> depth_processing:=false


- View image from camera in its own window:
rosrun rqt_image_view rqt_image_view


- Downloading freenect:
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install git-core cmake freeglut3-dev pkg-config build-essential libxmu-dev libxi-dev libusb-1.0-0-dev

cd ~/src
git clone https://github.com/OpenKinect/libfreenect.git
cd libfreenect
mkdir build
cd build
cmake -L ..
make
sudo make install
sudo ldconfig /usr/local/lib64/

cd ~/catkin_ws/src
git clone https://github.com/ros-drivers/freenect_stack.git
cd ..
catkin build
source ~/catkin_ws/devel/setup.bash


- Calibrating camera, follow guide below (install aruco_detect package first): 
- http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration


- Runnig aruco_detect:
- Add folder ros_demo (from this repository) to ~/catkin_ws/src
- Run catkin build inside the catkin_ws folder
roslaunch demo_ros aruco.launch


- Installing camera calibrator package:
sudo apt-get install ros-noetic-camera-calibrator
rosdep install camera_calibration


- Installing aruco_detect package:
sudo apt-get install ros-noetic-aruco-detect
rosdep install aruco_detect
