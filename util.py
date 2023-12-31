from djitellopy import Tello
import cv2 as cv
import time
def initTello():
    myDrone = Tello()
    #drone connection
    myDrone.connect()

    #set all speed to 0
    myDrone.for_back_velocity=0
    myDrone.left_right_velocity=0
    myDrone.up_down_velocity=0
    myDrone.yaw_velocity=0
    myDrone.speed=0
    print("\m Drone batterypercnetage : "+str(myDrone.get_battery())+"%")
    myDrone.streamoff()
    return myDrone

def moveTello(myDrone):
    myDrone.takeoff()
    time.sleep(5)

    myDrone.move_right(60)
    time.sleep(5)
    myDrone.move_left(60)
    time.sleep(5)

    myDrone.move_back(50)
    time.sleep(5)
    myDrone.rotate_clockwise(360)
    time.sleep(5)
    myDrone.move_forward(50)
    time.sleep(5)

    myDrone.land()
    time.sleep(5)
