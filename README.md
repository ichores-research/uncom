# UNCOM - Understanding of Commands

An algorithm for understanding human commands in table-top scenarios.

## Prerequisites

Download mediapipe model for the hand detection.

```bash
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -P models -o hand_landmarker.task
```

Install required packages. 

* If using conda you may try `environment.yml`.
* If using pip try `requirements.txt`

* Install ROS Noetic if you want to use the code in a real or simulated TIAGo robot. Follow this tutorial: http://wiki.ros.org/noetic/Installation/Ubuntu

Install `ffmpeg` (If using conda this may have alredy been done).

Create a ROS workspace with the following commands:

```bash
mkdir -p ~/uncom_ws/src  # creates workspace folder, you can change the folder name if desired. You can skip this step if you already have a workspace
cd ~/uncom_ws  # you can skip this step if you already have a workspace
catkin init  # you can skip this step if you already have a workspace
source devel/setup.bash
cp -a <location of the repository>/uncom/ ~/uncom_ws/src/uncom
catkin build
source devel/setup.bash
```

## Running

### Understanding only, no robot:

The main program expects a video in MP4 format consisting of one command involving one object, one action, and one target. Invoke the program as follows:

```bash
Usage:
python understand_real_time_PC.py -o [OUTPUT_DIRECTORY] [INPUT_VIDEO] [--device DEVICE]

Example:
python understand.py -o output/small_orange data/put_small_orange_in_bowl.mp4 
```

### ROS Node launch

```bash
cd ~/uncom_ws  # or the location of your actual catkin workspace
souce devel/setup.bash
roslaunch uncom uncom.launch
```

The program will output both end and intermediate results to the specified output directory. By default it will choose to use GPU if available. You can change that using the `--device` argument.

## Code organisation

* The main program `understand_real_time_PC.py` in in the top level directory.
* Python scripts for the two ROS nodes are in the /src folder. 
* Supporting files are in `src/uncom_utils` subfolder.
* `model_tests` contains ad-hoc tests of different models, before they get intergated into the main program.
