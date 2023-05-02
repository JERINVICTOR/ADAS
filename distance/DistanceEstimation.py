import cv2 as cv 
import numpy as np
import threading
from playsound import playsound # for audio alert
import os # for piping

import setproctitle # to set process name

setproctitle.setproctitle("Collision_alert_python")

def play_alert():
    # Play the alert sound asynchronously
    playsound('crashAlert.mp3', block=False)

# Distance constants 
KNOWN_DISTANCE = 1.143 #meter
PERSON_WIDTH = 0.4064 #meter
CAR_WIDTH = 1.7526
TRUCK_WIDTH = 2.5908
# Object detector constant 
CONFIDENCE_THRESHOLD = 0.7
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file 
class_names = []
with open("distance/classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('distance/yolov4-tiny.weights', 'distance/yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list =[]
    print(classes)
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id 
        color= COLORS[int(classid) % len(COLORS)]
    
        label = "%s : %f" % (class_names[classid[0]], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid ==0: # person class id 
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid==2:
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid==7:
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data. 
    return data_list

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

# reading the reference image from dir 
ref_person = cv.imread('distance/ReferenceImages/image14.png')

ref_car = cv.imread('distance/ReferenceImages/car2.png')
ref_truck = cv.imread('distance/ReferenceImages/truckA.png')


print("car")
car_data = object_detector(ref_car)
car_width_in_rf = car_data[0][1]

print("reached truck")
truck_data = object_detector(ref_truck)
truck_width_in_rf = truck_data[0][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]


# finding focal length 
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

focal_car = focal_length_finder(5, CAR_WIDTH, car_width_in_rf)

focal_truck = focal_length_finder(2, TRUCK_WIDTH, truck_width_in_rf)

cap = cv.VideoCapture(2)
person_thresh = 2.016
car_thresh = 3
truck_thresh = 7
bus_thresh = 1

# Open the named pipe for reading
pipe_path = 'lanePipe'
pipe = open(pipe_path, 'rb')

while True:
    #need to replace reading frame with piped frame here
    #ret, frame = cap.read()

    # Read the next frame from the named pipe
    frame_bytes = pipe.read(2764800)  # 720x1280x3 bytes
    if len(frame_bytes) != 2764800:
        continue

    # Convert the frame bytes to a numpy array
    try:
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((720, 1280, 3))
    except ValueError as e:
        print(f"Error constructing frame from buffer: {e}")

    

    data = object_detector(frame) 
    for d in data:
        if d[0] =='person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
            print("distance", distance)
            if distance < person_thresh:
                alert_thread = threading.Thread(target=play_alert)
                alert_thread.start()

        elif d[0] == 'car':
            distance = distance_finder (focal_car, CAR_WIDTH, d[1])
            x, y = d[2]
            if distance < car_thresh:
                alert_thread = threading.Thread(target=play_alert)
                alert_thread.start()
        elif d[0] == 'truck':
            distance = distance_finder (focal_truck, CAR_WIDTH, d[1])
            x, y = d[2]
            if distance < truck_thresh:
                alert_thread = threading.Thread(target=play_alert)
                alert_thread.start()
        elif d[0] == 'bus':
            distance = distance_finder (focal_truck, CAR_WIDTH, d[1])
            x, y = d[2] 
            if distance < bus_thresh:
                alert_thread = threading.Thread(target=play_alert)
                alert_thread.start()   
        cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
        cv.putText(frame, f'Dis: {round(distance,2)} meter', (x+5,y+13), FONTS, 0.48, GREEN, 2)

    cv.imshow('frame',frame)
    
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
pipe.close()
#cap.release()

