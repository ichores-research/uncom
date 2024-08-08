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

🤞

Install `ffmpeg` (If using conda this may have alredy been done).

## Running

The main program expects a video in MP4 format consisting of one command involving one object, one action, and one target. Invoke the program as follows:

```bash
Usage:
python understand.py -o [OUTPUT_DIRECTORY] [INPUT_VIDEO] [--device DEVICE]

Example:
python understand.py -o output/small_orange data/put_small_orange_in_bowl.mp4 
```

The program will output both end and intermediate results to the specified output directory. By default it will choose to use GPU if available. You can change that using the `--device` argument.

## Code organisation

* The main program `understand.py` in in the top level directory.
* Supporting files are in `uncom` subfolder.
* `model_tests` contains ad-hoc tests of different models, before they get intergated into the main program.
