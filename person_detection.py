from darkflow.net.build import TFNet
import cv2
import threading

# Darkflow
options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
tfnet = TFNet(options)

# Camera
camera = cv2.VideoCapture(0)
camera.set(3, 800)
camera.set(4, 800)

face_cascade = cv2.CascadeClassifier("bin/haarcascade_frontalface_default.xml")

current_frame = None
resp = None


def darkflow():
    result = tfnet.return_predict(current_frame)
    final_result = sorted(result, key=lambda k:(k['label']=='person', k['confidence']), reverse=True)
    print(final_result)
    global resp
    resp = final_result

def initial_camera():
    _, frame = camera.read()
    global current_frame
    current_frame = frame
    thread = threading.Thread(target=darkflow)
    thread.start()

    while(True):
        _, frame = camera.read()

        if thread.is_alive():
            pass
        else:
            current_frame = frame
            thread = threading.Thread(target=darkflow)
            thread.start()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Display the resulting frame
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            #  roi_gray = gray[y:y+h, x:x+w]
            #  roi_color = frame[y:y+h, x:x+w]
            #  face_capture = frame[y:y+h, x:x+w]

            # Draw text background
            cv2.rectangle(frame,
                (x - 1, y - 2),
                (x + 80, y - 35),
                (255, 0, 0),
                -1)

            # Draw face label
            cv2.putText(img=frame,
                text="face",
                org=(x + 10, y - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA)

        if resp:
            for i in resp:
                if i['confidence'] > 0.6: 
                    # Draw rectangle
                    cv2.rectangle(frame,
                        (i["topleft"]["x"], i["topleft"]["y"]),
                        (i["bottomright"]["x"], i["bottomright"]["y"]),
                        (0, 255, 0),
                        2)

                    # Draw text background
                    cv2.rectangle(frame,
                        (i["topleft"]["x"] - 1, i["topleft"]["y"] - 2),
                        (i["topleft"]["x"] + 180, i["topleft"]["y"] - 38),
                        (0, 255, 0),
                        -1)
                    
                    # Draw text label
                    text_x, text_y = i["topleft"]["x"], i["topleft"]["y"] - 12    
                    cv2.putText(img=frame,
                        text=i["label"] + ' ' + str(round(i["confidence"], 2)),
                        org=(text_x, text_y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color=(0, 0, 0),
                        thickness=2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    initial_camera()
