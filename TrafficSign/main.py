import cv2
import supervision as sv
from ultralytics import YOLO
from PIL import Image
import time
import setproctitle # to set process name

setproctitle.setproctitle("SignDetection_python")

def main():
    
    # to save the video

    # define resolution
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # specify the model
    model = YOLO("./TrafficSign/yoloSign.pt")

    # customize the bounding box
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )


    while True:
        ret, frame = cap.read()
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )


       
       
        #showing image output to infotainment system
        sign_list=["stop","crosswalk", "do_not_enter", "speed_limit_30", "speed_limit_60", "u_turn" ,"bus_stop"]
        for signs in sign_list:
            if(len(labels)!=0):
                if(labels[0].__contains__(signs)):
                    im = cv2.imread(f"TrafficSign/images/{signs}.jpg",cv2.IMREAD_ANYCOLOR)
                    if im is not None:
                        rimg = cv2.resize(im , (300,300))
                        cv2.imshow("Sign",rimg)
                    else:
                        print("Failed to load image")

                    
                    
            
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27): # break with escape key
            break
            
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
