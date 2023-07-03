import sys
from util import *
import cv2 as cv
import time
import keyboard
import pyttsx3
from threading import Thread
keepRecording=True
recorder=0
sys.path.append('pingpong')
from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp
import numpy as np
engine=pyttsx3.init()
engine.setProperty('rate',200)
x=0
y=0
actions = ['come', 'away', 'spin','left','right','up','down']
seq_length = 30

model = load_model('models/model_h.h5')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

seq = []
action_seq = []
last_action = None

if __name__ == '__main__':
    myDrone=initTello()

    # myDrone.takeoff()
    # time.sleep(1)
    myDrone.streamon()
    cv.namedWindow("drone")
    frame_read=myDrone.get_frame_read()
    time.sleep(2)
    while True:
        img = frame_read.frame
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if keyboard.is_pressed('q'):
            print('Q')
            myDrone.land()
            frame_read.stop()
            myDrone.streamoff()
            exit(0)
            break
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
                v = v2 - v1  # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                y_pred = model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.9:
                    continue

                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 3:
                    continue

                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action

                    if last_action != this_action:
                        if this_action == 'up':
                            print("up")
                            myDrone.takeoff()
                            time.sleep(1)
                            engine.say('이륙')
                            engine.runAndWait()
                        elif this_action == 'down':
                            print("down")
                            myDrone.land()
                            engine.say('착륙')
                            engine.runAndWait()
                        elif this_action == 'come':
                            print("come")
                            if x+60<=110:
                                x+=60
                                myDrone.move_forward(60)
                                engine.say('전진')
                                engine.runAndWait()
                            else:
                                print('범위에서 바캍입니다.')
                                engine.say('위험')
                                engine.runAndWait()
                        elif this_action == 'away':
                            print("away")
                            if x-60>-110:
                                x-=60
                                myDrone.move_back(60)
                                engine.say('후진')
                                engine.runAndWait()
                            else:
                                print('범위에서 바캍입니다.')
                                engine.say('위험')
                                engine.runAndWait()
                        elif this_action == 'spin':
                            print("spin")
                            myDrone.rotate_clockwise(360)
                            engine.say('회전')
                            engine.runAndWait()
                        elif this_action == 'left':
                            print('left')
                            if y + 60 <= 110:
                                y+=60
                                myDrone.move_right(60)
                                engine.say('좌로이동')
                                engine.runAndWait()
                            else:
                                print('범위에서 바캍입니다.')
                                engine.say('위험')
                                engine.runAndWait()
                        elif this_action == 'right':
                            print('right')
                            if y-60>-110:
                                y-=60
                                myDrone.move_left(60)
                                engine.say('우로이동')
                                engine.runAndWait()
                            else:
                                print('범위에서 바캍입니다.')
                                engine.say('위험')
                                engine.runAndWait()
                        last_action = this_action

                # cv2.putText(img, f'{this_action.upper()}',
                #             org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        cv.waitKey(80)

