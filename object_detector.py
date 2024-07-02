import cv2
import numpy as np
from scipy.spatial import distance as dist

# Constants
thres = 0.45  # Threshold to detect object
nms_threshold = 0.2  # (0.1 to 1) 1 means no suppress, 0.1 means high suppress
Known_distance = 30  # Inches
Known_width = 5.7  # Inches

# Colors (BGR Format)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 242)
GOLDEN = (32, 218, 165)
LIGHT_BLUE = (255, 9, 2)
PURPLE = (128, 0, 128)
CHOCOLATE = (30, 105, 210)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)

# Font settings
font = cv2.FONT_HERSHEY_PLAIN
fonts = cv2.FONT_HERSHEY_COMPLEX

# Load class names
classNames = []
classFile ='coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load configuration and weights
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Set up the network
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)  # Set height
cap.set(10, 70)  # Set brightness

# Face detector setup
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Focal length calculation
def FocalLength(measured_distance, real_width, width_in_rf_image):
    return (width_in_rf_image * measured_distance) / real_width

# Distance estimation function
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    return (real_face_width * Focal_Length) / face_width_in_frame

# Face detection function
def face_data(image, CallOut, Distance_level):
    face_width = 0
    face_center_x = 0
    face_center_y = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, w, h) in faces:
        if CallOut:
            cv2.putText(image, f"Distance {Distance_level} Inches", (x, y - 10), fonts, 0.5, (BLACK), 2)
        face_width = w
        face_center_x = int(w / 2) + x
        face_center_y = int(h / 2) + y
    return face_width, faces, face_center_x, face_center_y

# Reading reference image for focal length calculation
ref_image_path = 'lena.png'
ref_image = cv2.imread(ref_image_path)
if ref_image is None:
    print(f"Error: Unable to load reference image at {ref_image_path}")
    exit()

ref_image_face_width, _, _, _ = face_data(ref_image, False, 0)
Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Unable to capture frame from camera")
        break

    classIds, confs, bbox = net.detect(frame, confThreshold=thres)

    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    face_width_in_frame, Faces, FC_X, FC_Y = face_data(frame, True, 0)

    if len(classIds) != 0:
        for i in indices:
            box = bbox[i]
            confidence = str(round(confs[i], 2))
            color = (0, 255, 0)
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
            cv2.putText(frame, classNames[classIds[i] - 1] + " " + confidence, (x + 10, y + 20), font, 1, color, 2)
            size_text = f'Size: {w}x{h}'
            cv2.putText(frame, size_text, (x + 10, y + 60), font, 1, color, 2)

    if face_width_in_frame != 0:
        Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)
        Distance = round(Distance, 2)
        cv2.putText(frame, f"Distance {Distance} Inches", (50, 50), fonts, 0.75, (BLACK), 2)

    if cv2.waitKey(1) == ord('q'):
        break

    cv2.imshow("Output", frame)

cap.release()
cv2.destroyAllWindows()
