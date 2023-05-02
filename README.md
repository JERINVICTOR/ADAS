# Advanced-Driver-Alertness-System

download shape_predictor_68_face_landmarks.dat before running drows.py
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

# Traffic-Detection

download yolov8 and add it to path
( or )
```bash
pip install ultralytics
```

run:
```bash
yolo task=detect mode=predict model='yoloSign.pt' conf=0.80 source=0 show=True
```

# UI integration

Install Anaconda . Then create required enviroments from CondaEnv.

```bash
conda env create -f laneTransformEnv.yml
```
```bash
conda env create -f SignEnv.yml
```
```bash
conda env create -f drowsyEnv.yml
```
```bash
conda env create -f DistanceEnv.yml
```
```bash
conda env create -f ADAS_UI_env.yml
```
Activate ADAS conda Enviroment by running 

```bash
conda activate ADAS
```
Run the UI
```bash
python processcontrollui.py
```
# Deep learning based lane detection

Install Anaconda then use the lane_environment.yml file to create the environment

```bash
conda env create -f lane_environment.yml
```
activate environment - 
Linux -
```bash
source activate lane_environment
```
Windows -
```bash
conda activate lane_environment
```
