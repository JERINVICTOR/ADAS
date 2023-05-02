import numpy as np
import cv2
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import load_model
import time
import tensorflow as tf
import setproctitle # to set process name

setproctitle.setproctitle("ANN_lane_python")
frame_count = 0
start_time = time.time()
# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """

    # Get image ready for feeding into model
    small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = imresize(lane_drawn, (720, 1280, 3))
    lane_image= cv2.resize(lane_image, (image.shape[1], image.shape[0]))
    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result

# tf.keras.backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)))
if __name__ == '__main__':
    # Load Keras model
    model = load_model('nnlanedetection/full_CNN_model.h5')
    # Create lanes object
    lanes = Lanes()
    
    cap = cv2.VideoCapture(2)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = road_lines(frame)
        cv2.imshow("lane detection", result)
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        print(fps)

        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break

    
