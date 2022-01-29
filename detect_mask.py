
import argparse
import os
import time
import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import datetime




def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)

            # face = preprocess_input(face)
            # face = np.expand_dims(face, axis=0)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="train",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="train/mask_detector1k.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())



# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = cv2.VideoCapture(0)
time.sleep(2.0)



prev_frame_time = 0
realtime = 1
f_xmin = open("xmin.txt","w+", realtime)
f_ymin = open("ymin.txt","w+", realtime)
f_xmax = open("xmax.txt","w+", realtime)
f_ymax = open("ymax.txt","w+", realtime)

startX= 0
startY = 0
endX = 0
endY = 0

c = 0
# loop over the frames from the video stream
while True:
    
    new_frame_time = 0
    ret,frame = vs.read()

    
    frame = imutils.resize(frame, width=500)
    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    


    if c % 40 == 0:
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            
            outputxmin = startX
            outputymin = startY
            outputxmax = endX
            outputymax = endY
            realtime = 1

            for i in range(10):
                f_xmin.write( "%s %s \n" %(str(outputxmin),datetime.datetime.now() ))
                f_xmin.flush()

                f_ymin.write("%s %s \n" %(str(outputymin),datetime.datetime.now() ))
                f_ymin.flush()

                f_xmax.write("%s %s \n" %(str(outputxmax),datetime.datetime.now() ))
                f_xmax.flush()

                f_ymax.write("%s %s \n" %(str(outputymax),datetime.datetime.now() ))
                f_ymax.flush()




            if(withoutMask > 0.10):
                withoutMask = 1
            # #     # time.sleep(2)
        


            # withoutMask = withoutMask/2
            # mask = mask * 3
            print("mask" , mask , "WMAsk" , withoutMask)


            if (mask > withoutMask):
                label = "Thank You. Mask On."
                color = (0, 255, 0)
                cv2.putText(frame, label, (startX-50, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.imshow("asd", frame)


            else:
                label = "No Face Mask Detected"
                color = (0, 0, 255)
                cv2.putText(frame, label, (startX-50, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.imshow("asd", frame)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)




    c += 1


    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, str(int(fps)), (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Face Mask Detector", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup


f_xmin.close()
f_ymin.close()
f_xmax.close()
f_ymax.close()


vs.release()
cv2.destroyAllWindows()






