from aws_index_face import upload_index_faces

import argparse
import cv2
import numpy as np

# Argument parser
parser = argparse.ArgumentParser(description='Name of saving image')
parser.add_argument("-n", "--name", required=True, help="Name of saving image", type=str)
parser.add_argument("-u", "--upload", required=False, help="Upload to S3", type=bool, default=False)
args = vars(parser.parse_args())

# Camera initialization
camera = cv2.VideoCapture(0)
camera.set(3, 800)
camera.set(4, 800)

# Caffe network initialization
caffe_net = cv2.dnn.readNetFromCaffe('bin/deploy.prototxt.txt', 'bin/res10_300x300_ssd_iter_140000.caffemodel')

def capture_image():
    """Capture an image to save in local and upload to S3
    """

    printed = True
    while True:
        _, frame = camera.read()

        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))

        caffe_net.setInput(blob)
        detections = caffe_net.forward()

        faces_number = 1
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < 0.5:
                continue

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")

            # draw the bounding box of the face along with the associated
            # probability
            text = "Face" + str(faces_number) + " {:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            # Draw rectangle
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (255, 0, 0), 2)

            # Draw background
            cv2.rectangle(frame,
                (startX - 1, startY - 2),
                (startX + 200, startY - 35),
                (255, 0, 0),
                -1)

            # Draw text
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # Increasing faces number of detection
            faces_number += 1

        cv2.imshow('frame', frame)

        if printed:
            print('"Press Q to capture image"')
            printed = False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(args["name"] + '.jpg', frame)
            if args["upload"]:
                upload_index_faces(args["name"] + '.jpg')
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()
