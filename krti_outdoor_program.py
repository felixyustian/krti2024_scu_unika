# define the required libraries
from __future__ import print_function
import cv2
import torch
import time, serial, sys, argparse, math, socket, pickle, os, struct
import pandas as pd
import numpy as np
from dronekit import connect, Command, LocationGlobal, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import socket, sys, pickle, struct
import imagezmq
import subprocess as sp
import ffmpeg
import datetime
from threading import Thread
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import traceback
import keyboard
import csv
from gpiozero import DistanceSensor, LineSensor
import RPi.GPIO as GPIO
from adafruit_servokit import ServoKit
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
from mjpeg_streamer import MjpegServer, Stream

# define the detected object class and confidence threshold
object_detected_1 = "payload"
object_detected_2 = "box"
object_detected_3 = "window"
left_landing_zone_detected = "red"
right_landing_zone_detected = "blue"
payload_outdoor_zone_detected = "white"
threshold = 0.2
max_detect_object = 1
# select the best model for the the object detection
model_path = 'best_batch10_epochs150.pt'
# model_path = 'best_batch20_epochs150.pt'
# model_path = 'best_batch10_epochs200.pt'
# model_path = 'best_batch20_epochs200.pt'

# define the input camera source port number
bottom_camera = 0
front_camera = 2

# define the safe distance limit of the robot for the HC-SR04 ultrasonic sensors in centimeter(s)
distance_front_minimum = 80.00
distance_left_minimum = 80.00
distance_right_minimum = 80.00
distance_back_minimum = 80.00

# initialize the known or reuuired distance from the camera to the object, which in this case is 800 mm
known_distance = 800.00
# initialize the known object width, which in this case, the diameter of the obstacle pole is 900 mm
known_width = 900.00
# set the safe distance between from the camera to surrounding object(s) in milimeters
safe_distances = 800.00

# define the waypoint numbers for the AUTO mission
# indoor mission
home_waypoint = 0
home_loiter_waypoint = 1
payload_waypoint = 2
payload_loiter_waypoint = 3
forward_corner_waypoint = 4
corner_loiter_waypoint = 5
corner_forward_waypoint = 6
drop_waypoint = 7
drop_loiter_waypoint = 8
window_loiter_waypoint = 9
window_forward_waypoint_before_indoor = 10
window_forward_waypoint_before_outdoor = 10
window_forward_waypoint_after = 11
window_loiter_waypoint_after = 12
# outdoor mission
outdoor_drop_mission_position_1_forward_waypoint_left = 13
outdoor_drop_mission_position_1_loiter_waypoint_left = 14
outdoor_drop_mission_position_2_forward_waypoint_left = 15
outdoor_drop_mission_position_2_loiter_waypoint_left = 16
outdoor_drop_mission_position_1_forward_waypoint_right = 17
outdoor_drop_mission_position_1_loiter_waypoint_right = 18
outdoor_drop_mission_position_2_forward_waypoint_right = 19
outdoor_drop_mission_position_2_loiter_waypoint_right = 20
going_new_land_loiter_waypoint_left = 21
going_new_land_loiter_waypoint_right = 22
going_new_land_forward_waypoint_left = 22
going_new_land_forward_waypoint_right = 23
new_land_loiter_waypoint_left = 24
new_land_loiter_waypoint_right = 25

# define the servo's angle for the gripper and front camera bracket servos
gripper_left_servo_angle_start = 90
gripper_left_servo_angle_final = 95
gripper_left_servo_angle_release = 45
gripper_right_servo_angle_start = 180
gripper_right_servo_angle_final = 175
gripper_right_servo_angle_release = 45
front_cam_servo_angle_front = 90
front_cam_servo_angle_bottom = 180
payload_drop_servo_left_start = 0
payload_drop_servo_left_final = 180
payload_drop_servo_right_start = 0
payload_drop_servo_right_final = 180

# define altitude for indoor mission in meter(s)
altitude_indoor = 0.5
altitude_drop = 1.5
altitude_sensor_placement = 0.1

# define altitude for outdoor mission in meter(s)
altitude_outdoor = 5

# define the waiting time for the object detection system to get ready in seconds
wait_time = 10

# define the ip address and port of the localhost for streaming
localhost = "192.168.0.110" # change the ip address of the raspberry pi on the network, same as ssh ip address
port_front = 8080
port_bottom = 8081

# define the duration for the vehicle movements in seconds
forward_duration_payload = 2 # forward to the payload position
forward_duration_box_1 = 6 # forward to the corner of box position
forward_duration_box_2 = 8 # forward to the box position
forward_duration_window = 5 # forward to the window position
forward_duration_new_home_1 = 3 # forward to the corner of new home position
forward_duration_new_home_2 = 0 # forward to the new home position
hold_duration = 10 # hold the position of the vehicle
return_duration = 6 # duration for going back to home (if error happened)

# define the vehicle angles for the mission
yaw_angle_left = 45
yaw_angle_right = 334
# yaw_angle_left_going_home_1 = 0
yaw_angle_right_going_home_1 = 67
# yaw_angle_left_going_home_2 = 0
# yaw_angle_left_going_home_3 = 0
# yaw_angle_right_going_home_3 = 0
start_heading = 243 # set the start heading
deviation_heading = 0
target_roll = 0
target_pitch = 0
deviation_roll = 0

# define the vehicle movement left or right (activate only for indoor and outdoor mission combination)
yaw_angle_option = -1 # going left
# yaw_angle_option = 1 # going right

# define the PWM frequency output
pwm_frequency = 500 # in Hertz
pwm_duty_cycle = 100 # in percent (0-100)

# define the cropping window size and fps for the object detection system
crop_window_size_width = 320
crop_window_size_height = 320
crop_window_fps = 10

# define the connection string and baudrate for connection to the arduino
# connection_string_serial = '/dev/ttyAMA0'
# connection_string_serial = '/dev/ttyACM1'
# connection_string_serial = '/dev/ttyACM2'
# connection_string_serial = '/dev/ttyACM3'
# connection_string_serial = '/dev/ttyACM4'
# baudrate_serial = 115200

# define RaspberryPi's GPIO pins for left and right line detection sensor
GPIO.setmode(GPIO.BCM)			    # GPIO numbering
GPIO.setwarnings(False)			        # enable warning from GPIO

# define the connection string for connection to the vehicle
connection_string_vehicle = '/dev/ttyAMA0, 115200' # port address and baudrate
# connection_string_vehicle = '/dev/ttyAMA1, 115200' # port address and baudrate
# connection_string_vehicle = '/dev/ttyAMA2, 115200' # port address and baudrate
# connection_string_vehicle = '/dev/ttyAMA3, 115200' # port address and baudrate
# connection_string_vehicle = '/dev/ttyAMA4, 115200' # port address and baudrate

# OpenGrab EPM v3 Magnetic Gripper
# define the Raspberry Pi's pin for the magnetic gripper
magnetic_gripper_pin = 13 # pin 33
GPIO.setup(magnetic_gripper_pin, GPIO.OUT)
# activate the GPIO PWM frequency output
magnetic_gripper = GPIO.PWM(magnetic_gripper_pin, pwm_frequency) # activate the PWM output for the magnetic gripper
magnetic_gripper.start(0) # start the PWM at duty cycle 0 percent

# HC-SR04 ultrasonic sensor
# define the ultrasonic sensor-based distance sensor Raspberry Pin's input pins
# trigger pins
distance_sensor_front_trigger = 17 # pin 11
distance_sensor_left_trigger = 27 # pin 13
distance_sensor_right_trigger = 22 # pin 15
distance_sensor_back_trigger = 4 # pin 7
# altitude_sensor_trigger = 7 # pin 26
# echo pins
distance_sensor_front_echo = 23 # pin 16
distance_sensor_left_echo = 24 # pin 18
distance_sensor_right_echo = 25 # pin 22
distance_sensor_back_echo = 8 # pin 24
# altitude_sensor_echo = 1 # pin 28
# calling the DistanceSensor function from gpiozero library
sensor_ping_front = DistanceSensor(trigger=distance_sensor_front_trigger, echo=distance_sensor_front_echo)
sensor_ping_left = DistanceSensor(trigger=distance_sensor_left_trigger, echo=distance_sensor_left_echo)
sensor_ping_right = DistanceSensor(trigger=distance_sensor_right_trigger, echo=distance_sensor_right_echo)
sensor_ping_back = DistanceSensor(trigger=distance_sensor_back_trigger, echo=distance_sensor_back_echo)
# sensor_ping_altitude = DistanceSensor(trigger=altitude_sensor_trigger, echo=altitude_sensor_echo)

# define the Raspberry Pi pin for downward line sensor for detect the payload
line_sensor = LineSensor(26) # pin 37
if line_sensor == 1:
    print("\n")
    print("Indoor payload not detected")
    print("\n")
if line_sensor == 0:
    print("\n")
    print("Indoor payload is detected")
    print("\n")

# Adafruit Servo Driver Hat
# define the Raspberry Pi I2C pin input for the Adafruit Servo driver
servo_kit = ServoKit(channels=16)
# define the servo channel on the driver
servo_gripper_left_channel = 7
servo_gripper_right_channel = 1
servo_front_cam_channel = 2
servo_payload_left_channel = 3
servo_payload_right_channel = 4
# define the channels of the servos on the driver
servo_gripper_left = servo_kit.servo[servo_gripper_left_channel]
servo_gripper_right = servo_kit.servo[servo_gripper_right_channel]
servo_gripper_front_cam = servo_kit.servo[servo_front_cam_channel]
servo_payload_left = servo_kit.servo[servo_payload_left_channel]
servo_payload_right = servo_kit.servo[servo_payload_right_channel]

# define the TCP address for sending the images
# tcp_address = 'tcp://192.168.0.106:5555'

class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self
	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()
	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()

class WebcamVideoStream:
	def __init__(self, src=0):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False
	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self
	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return
			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()
	def read(self):
		# return the frame most recently read
		return self.frame
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=float('inf'),
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

def wait_key():
    print("\n")
    print("Press the required key below to continue or exit the mission !!!")
    print("\n")
    print("1.   q or w  : to continue the mission")
    print("2.   a or s  : to exit the mission")
    print("\n")
    print("Press Ctrl + C if there is any problem during the mission to exit the mission !!!")
    print("\n")
    key_input = input("Enter the required key and press Enter:   ")

    while True:
        if key_input == "q" or key_input == "w":
            print("\n")
            print("[INFO] Key q or w is pressed...")
            print("\n")
            print("[INFO] Continue to the mission...")
            print("\n")
            main_code()
        if key_input == "a" or key_input == "s":
            print("\n")
            print("[INFO] Key a or s is pressed...")
            print("\n")
            print("[INFO] Exiting the mission...")
            print("\n")
            exit(0)
        else:
            print("\n")
            print("[INFO] Required key not detected...")
            print("\n")
            print("[INFO] Exiting the mission...")
            print("\n")
            exit(0)

# ====================================================================================================================================================================================== #


print("====================================================================================================================================================")
print("\n")
print("[INFO] SYSTEM IS STARTING...")
print("\n")

print("\n")
print('[INFO] SYSTEM IS RUNNING...')
print("\n")

# serial = serial.Serial(connection_string_serial, baudrate_serial, timeout=1)
# serial.flush()
# if serial.isOpen():
#     print("\n")
#     print("[INFO] ARDUINO VIA {} CONNECTED!".format(serial.port))
#     print("\n")

# connect to the vehicle via dronekit
# set `wait_ready=True` to ensure default attributes are populated before `connect()` returns.
print("\n[INFO] CONNECTING THE VEHICLE ON: %s" % connection_string_vehicle)

vehicle = connect(connection_string_vehicle, wait_ready=True)

vehicle.wait_ready('autopilot_version')

print("\n")
print("[INFO] LOADING THE VEHICLE PARAMETERS...")
print("\n")

# Get all vehicle attributes (state)
print("====================================================================================================================================================")
print("\n[INFO] Get all vehicle attribute values:")
print(" Autopilot Firmware version: %s" % vehicle.version)
print("   Major version number: %s" % vehicle.version.major)
print("   Minor version number: %s" % vehicle.version.minor)
print("   Patch version number: %s" % vehicle.version.patch)
print("   Release type: %s" % vehicle.version.release_type())
print("   Release version: %s" % vehicle.version.release_version())
print("   Stable release?: %s" % vehicle.version.is_stable())
print(" Autopilot capabilities")
print("   Supports MISSION_FLOAT message type: %s" % vehicle.capabilities.mission_float)
print("   Supports PARAM_FLOAT message type: %s" % vehicle.capabilities.param_float)
print("   Supports MISSION_INT message type: %s" % vehicle.capabilities.mission_int)
print("   Supports COMMAND_INT message type: %s" % vehicle.capabilities.command_int)
print("   Supports PARAM_UNION message type: %s" % vehicle.capabilities.param_union)
print("   Supports ftp for file transfers: %s" % vehicle.capabilities.ftp)
print("   Supports commanding attitude offboard: %s" % vehicle.capabilities.set_attitude_target)
print("   Supports commanding position and velocity targets in local NED frame: %s" % vehicle.capabilities.set_attitude_target_local_ned)
print("   Supports set position + velocity targets in global scaled integers: %s" % vehicle.capabilities.set_altitude_target_global_int)
print("   Supports terrain protocol / data handling: %s" % vehicle.capabilities.terrain)
print("   Supports direct actuator control: %s" % vehicle.capabilities.set_actuator_target)
print("   Supports the flight termination command: %s" % vehicle.capabilities.flight_termination)
print("   Supports mission_float message type: %s" % vehicle.capabilities.mission_float)
print("   Supports onboard compass calibration: %s" % vehicle.capabilities.compass_calibration)
print(" Global Location: %s" % vehicle.location.global_frame)
print(" Global Location (relative altitude): %s" % vehicle.location.global_relative_frame)
print(" Local Location: %s" % vehicle.location.local_frame)
print(" Attitude: %s" % vehicle.attitude)
print(" Velocity: %s" % vehicle.velocity)
print(" GPS: %s" % vehicle.gps_0)
print(" Gimbal status: %s" % vehicle.gimbal)
print(" Battery: %s" % vehicle.battery)
print(" EKF OK?: %s" % vehicle.ekf_ok)
print(" Last Heartbeat: %s" % vehicle.last_heartbeat)
print(" Rangefinder: %s" % vehicle.rangefinder)
print(" Rangefinder distance: %s" % vehicle.rangefinder.distance)
print(" Rangefinder voltage: %s" % vehicle.rangefinder.voltage)
print(" Heading: %s" % vehicle.heading)
print(" Is Armable?: %s" % vehicle.is_armable)
print(" System status: %s" % vehicle.system_status.state)
print(" Groundspeed: %s" % vehicle.groundspeed)    # settable
print(" Airspeed: %s" % vehicle.airspeed)    # settable
print(" Mode: %s" % vehicle.mode.name)    # settable
print(" Armed: %s" % vehicle.armed)    # settable
print("====================================================================================================================================================")

# ==================================================================================================================================================== # 

# Mission parameters and coding #

# ==================================================================================================================================================== # 

# get the distance sensor data in centimeter(s) (multiply by 100) and round it in 2 decimal digits
# distance_altitude = round(sensor_ping_altitude.distance * 100, 2) # activate only if the HC-SR04 sensor facing down is installed
# set the minimum requirements for the dirance measurements based on HC-SR-04 ultrasonic sensor
# if distance_altitude >= altitude_sensor_placement:    
#     print("\n")
#     print("[INFO] Ultrasonic sensor-based altitude sensor is connected...")
#     print("\n")
#     print("[INFO] Vehicle altitude: ", distance_altitude, "meter(s)")
#     print("\n")
#     if distance_altitude >= altitude_indoor:
#         print("\n")
#         print("[INFO] Safe altitude for indoor mission")
#         print("\n")
#     if distance_altitude >= altitude_outdoor:
#         print("\n")
#         print("[INFO] Safe altitude for outdoor mission")
#         print("\n")
#     if distance_altitude < altitude_indoor:
#         print("\n")
#         print("[INFO] WARNING !!! DANGEROUS ALTITUDE FOR INDOOR MISSION")
#         print("\n")
#         vehicle.mode = VehicleMode("LAND")
#     if distance_altitude < altitude_outdoor:
#         print("\n")
#         print("[INFO] WARNING !!! DANGEROUS ALTITUDE FOR OUTDOOR MISSION")
#         print("\n")
# if distance_altitude < altitude_sensor_placement:    
#     print("\n")
#     print("[INFO] Ultrasonic sensor-based altitude sensor not connected...")
#     print("\n")
distance_front = round(sensor_ping_front.distance * 100, 2) 
print("\n")
print("[INFO] Distance to front object: ", distance_front, "meter(s)")
print("\n")
if distance_front >= distance_front_minimum:
    print("\n")
    print("[INFO] Safe distance to front object...")
    print("\n")
if distance_front < distance_front_minimum:
    print("\n")
    print("[INFO] DANGEROUS DISTANCE TO FRONT OBJECT... !!!")
    print("\n")
distance_left = round(sensor_ping_left.distance * 100, 2)
print("\n")
print("[INFO] Distance to left object: ", distance_left, "meter(s)")
print("\n")
if distance_left >= distance_left_minimum:
    print("\n")
    print("[INFO] Safe distance to left object...")
    print("\n")
if distance_left < distance_left_minimum:
    print("\n")
    print("[INFO] DANGEROUS DISTANCE TO LEFT OBJECT... !!!")
    print("\n")
distance_right = round(sensor_ping_right.distance * 100, 2)
print("\n")
print("[INFO] Distance to right object: ", distance_right, "meter(s)")
print("\n")
if distance_right >= distance_right_minimum:
    print("\n")
    print("[INFO] Safe distance to right object...")
    print("\n")
if distance_right < distance_right_minimum:
    print("\n")
    print("[INFO] DANGEROUS DISTANCE TO RIGHT OBJECT... !!!")
    print("\n")
distance_back = round(sensor_ping_back.distance * 100, 2)
print("\n")
print("[INFO] Distance to back object: ", distance_back, "meter(s)")
print("\n")
if distance_back >= distance_back_minimum:
    print("\n")
    print("[INFO] Safe distance to back object...")
    print("\n")
if distance_back < distance_back_minimum:
    print("\n")
    print("[INFO] DANGEROUS DISTANCE TO BACK OBJECT... !!!")
    print("\n")

# Get Vehicle Home location - will be `None` until first set by autopilot
while not vehicle.home_location:
    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready()
    if not vehicle.home_location:
        print("[INFO] Waiting for home location ...")
# Print the home location        
print("\n[INFO] Home location: %s" % vehicle.home_location)

# Set vehicle home_location, mode, and armed attributes (the only settable attributes)

print("\n[INFO] Set new home location")
# # Home location must be within 50km of EKF home location (or setting will fail silently)
# # In this case, just set value to current location with an easily recognisable altitude (0)
my_location_alt = vehicle.location.global_frame
my_location_alt.alt = 0
vehicle.home_location = my_location_alt
print("[INFO] New Home Location (from attribute - altitude should be 0): %s" % vehicle.home_location)

#Confirm current value on vehicle by re-downloading commands
cmds = vehicle.commands
cmds.download()
cmds.wait_ready()
print("[INFO] New Home Location (from vehicle - altitude should be 0): %s" % vehicle.home_location)

# set the vehicle mode to GUIDED_NOGPS
print("\n")
print("\n[INFO] Set Vehicle.mode = GUIDED (currently: %s)" % vehicle.mode.name)
print("\n")
vehicle.mode = VehicleMode("GUIDED")
while not vehicle.mode.name=='GUIDED':  #Wait until mode has changed
    print("\n")
    print(" Waiting for mode change ...")
    # print("[INFO] Current Vehicle Mode = " % vehicle.mode.name)
    # print("\n")
    time.sleep(1)

def arm_and_takeoff_nogps(aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude without GPS data.
    """

    ##### CONSTANTS #####
    DEFAULT_TAKEOFF_THRUST = 0.7
    SMOOTH_TAKEOFF_THRUST = 0.6

    print("\n")
    print("[INFO] Basic pre-arm checks - DON'T TOUCH !!!")
    print("\n")
    # Don't let the user try to arm until autopilot is ready
    # If you need to disable the arming check,
    # just comment it with your own responsibility.
    # while not vehicle.is_armable:
    #     print("\n")
    #     print("[INFO] Waiting for vehicle to initialise...")
    #     print("\n")
    #     time.sleep(1)
    # print("[INFO] Waiting RC Channel 6 / GEAR input to arming")
    # print("\n")
    # ch_6_input = vehicle.channels['6']
    # if ch_6_input > 1000:
    print("\n")
    print("[INFO] Arming motors")
    print("\n")
    # Copter should arm in GUIDED_NOGPS mode
    vehicle.mode = VehicleMode("GUIDED_NOGPS")
    print("\n")
    print("\n[INFO] Set Vehicle.mode = GUIDED_NOGPS (currently: %s)" % vehicle.mode.name)
    print("\n")
    vehicle.armed = True

    while not vehicle.armed:
        print("\n")
        print("[INFO] Waiting for arming...")
        print("\n")
        vehicle.armed = True
        time.sleep(1)
    
    print("\n")
    print("[INFO] Taking off!")
    print("\n")
    global thrust
    thrust = DEFAULT_TAKEOFF_THRUST
    target_heading = start_heading
    while True:
        current_heading = vehicle.heading
        current_roll = vehicle.attitude.roll
        print("[INFO] Current heading: ", current_heading)
        print("\n")
        if current_heading < target_heading:
           deviation_heading = target_heading - current_heading
           send_attitude_target(pitch_angle=0, yaw_angle=(current_heading + deviation_heading), thrust=SMOOTH_TAKEOFF_THRUST, duration=3)
        if current_heading > target_heading:
           deviation_heading = current_heading - target_heading
           send_attitude_target(pitch_angle=0, yaw_angle=(current_heading - deviation_heading), thrust=SMOOTH_TAKEOFF_THRUST, duration=3)
        if current_roll < target_roll:
           deviation_roll = target_roll - current_roll
           send_attitude_target(pitch_angle=0, roll_angle=(current_roll + deviation_roll), thrust=SMOOTH_TAKEOFF_THRUST, duration=3)
        if current_roll > target_roll:
           deviation_roll = current_roll - target_roll
           send_attitude_target(pitch_angle=0, roll_angle=(current_roll - deviation_roll), thrust=SMOOTH_TAKEOFF_THRUST, duration=3)
        current_altitude = vehicle.location.global_relative_frame.alt
        # sonar_alt = vehicle.rangefinder("distance")
        # current_altitude = sonar_alt
        print("\n")
        print("[INFO] Altitude: %f  Desired: %f" %
              (current_altitude, aTargetAltitude))
        print("\n")
        if current_altitude >= aTargetAltitude*0.95: # Trigger just below target alt.
            print("\n")
            print("[INFO] Reached target altitude")
            print("\n")
            break
        elif current_altitude >= aTargetAltitude*0.6:
            thrust = SMOOTH_TAKEOFF_THRUST
        send_attitude_target(thrust = thrust)
        time.sleep(0.2)

def send_attitude_target(roll_angle = 0.0, pitch_angle = 0.0,
                         yaw_angle = None, yaw_rate = 0.0, use_yaw_rate = False,
                         thrust = 0.5):
    """
    use_yaw_rate: the yaw can be controlled using yaw_angle OR yaw_rate.
                  When one is used, the other is ignored by Ardupilot.
    thrust: 0 <= thrust <= 1, as a fraction of maximum vertical thrust.
            Note that as of Copter 3.5, thrust = 0.5 triggers a special case in
            the code for maintaining current altitude.
    """
    if yaw_angle is None:
        # this value may be unused by the vehicle, depending on use_yaw_rate
        yaw_angle = vehicle.attitude.yaw
    # Thrust >  0.5: Ascend
    # Thrust == 0.5: Hold the altitude
    # Thrust <  0.5: Descend
    msg = vehicle.message_factory.set_attitude_target_encode(
        0, # time_boot_ms
        1, # Target system
        1, # Target component
        0b00000000 if use_yaw_rate else 0b00000100,
        to_quaternion(roll_angle, pitch_angle, yaw_angle), # Quaternion
        0, # Body roll rate in radian
        0, # Body pitch rate in radian
        math.radians(yaw_rate), # Body yaw rate in radian/second
        thrust  # Thrust
    )
    vehicle.send_mavlink(msg)

def set_attitude(roll_angle = 0.0, pitch_angle = 0.0,
                 yaw_angle = None, yaw_rate = 0.0, use_yaw_rate = False,
                 thrust = 0.5, duration = 0):
    """
    Note that from AC3.3 the message should be re-sent more often than every
    second, as an ATTITUDE_TARGET order has a timeout of 1s.
    In AC3.2.1 and earlier the specified attitude persists until it is canceled.
    The code below should work on either version.
    Sending the message multiple times is the recommended way.
    """
    send_attitude_target(roll_angle, pitch_angle,
                         yaw_angle, yaw_rate, False,
                         thrust)
    start = time.time()
    while time.time() - start < duration:
        send_attitude_target(roll_angle, pitch_angle,
                             yaw_angle, yaw_rate, False,
                             thrust)
        time.sleep(0.1)
    # Reset attitude, or it will persist for 1s more due to the timeout
    send_attitude_target(0, 0,
                         0, 0, True,
                         thrust)

def to_quaternion(roll = 0.0, pitch = 0.0, yaw = 0.0):
    """
    Convert degrees to quaternions
    """
    t0 = math.cos(math.radians(yaw * 0.5))
    t1 = math.sin(math.radians(yaw * 0.5))
    t2 = math.cos(math.radians(roll * 0.5))
    t3 = math.sin(math.radians(roll * 0.5))
    t4 = math.cos(math.radians(pitch * 0.5))
    t5 = math.sin(math.radians(pitch * 0.5))

    w = t0 * t2 * t4 + t1 * t3 * t5
    x = t0 * t3 * t4 - t1 * t2 * t5
    y = t0 * t2 * t5 + t1 * t3 * t4
    z = t1 * t2 * t4 - t0 * t3 * t5

    return [w, x, y, z]

def arm_and_takeoff(aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """
    
    print("\n")
    print("[INFO] Basic pre-arm checks")
    print("\n")
    # Don't let the user try to arm until autopilot is ready
    # while not vehicle.is_armable:
    #     print("\n")
    #     print("[INFO] Waiting for vehicle to initialise...")
    #     print("\n")
    #     time.sleep(1)

    print("\n")        
    print("[INFO] Arming motors")
    print("\n")
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    print("\n")
    print("\n[INFO] Set Vehicle.mode = GUIDED (currently: %s)" % vehicle.mode.name)
    print("\n")
    vehicle.armed = True

    while not vehicle.armed:
        print("\n")
        print("[INFO] Waiting for arming...")
        print("\n")
        time.sleep(1)
    
    print("\n")
    print("[INFO] Taking off!")
    print("\n")
    vehicle.simple_takeoff(aTargetAltitude) # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command 
    #  after Vehicle.simple_takeoff will execute immediately).
    while True:
        print("\n")
        print("[INFO] Altitude: ", vehicle.location.global_relative_frame.alt)
        print("\n")
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95: #Trigger just below target alt.
            print("\n")
            print("[INFO] Reached target altitude")
            print("\n")
            break
        time.sleep(1)

def condition_yaw(heading, relative=False):
    """
    Send MAV_CMD_CONDITION_YAW message to point vehicle at a specified heading (in degrees).

    This method sets an absolute heading by default, but you can set the `relative` parameter
    to `True` to set yaw relative to the current yaw heading.

    By default the yaw of the vehicle will follow the direction of travel. After setting 
    the yaw using this function there is no way to return to the default yaw "follow direction 
    of travel" behaviour (https://github.com/diydrones/ardupilot/issues/2427)

    For more information see: 
    http://copter.ardupilot.com/wiki/common-mavlink-mission-command-messages-mav_cmd/#mav_cmd_condition_yaw
    """
    if relative:
        is_relative = 1 #yaw relative to direction of travel
    else:
        is_relative = 0 #yaw is an absolute angle
    # create the CONDITION_YAW command using command_long_encode()
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
        0, #confirmation
        heading,    # param 1, yaw in degrees
        0,          # param 2, yaw speed deg/s
        1,          # param 3, direction -1 ccw, 1 cw
        is_relative, # param 4, relative offset 1, absolute angle 0
        0, 0, 0)    # param 5 ~ 7 not used
    # send command to vehicle
    vehicle.send_mavlink(msg)

def set_roi(location):
    """
    Send MAV_CMD_DO_SET_ROI message to point camera gimbal at a 
    specified region of interest (LocationGlobal).
    The vehicle may also turn to face the ROI.

    For more information see: 
    http://copter.ardupilot.com/common-mavlink-mission-command-messages-mav_cmd/#mav_cmd_do_set_roi
    """
    # create the MAV_CMD_DO_SET_ROI command
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_DO_SET_ROI, #command
        0, #confirmation
        0, 0, 0, 0, #params 1-4
        location.lat,
        location.lon,
        location.alt
        )
    # send command to vehicle
    vehicle.send_mavlink(msg)

def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the 
    specified `original_location`. The returned LocationGlobal has the same `alt` value
    as `original_location`.

    The function is useful when you want to move the vehicle around specifying locations relative to 
    the current vehicle position.

    The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.

    For more information see:
    http://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
    """
    earth_radius = 6378137.0 #Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    #New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    if type(original_location) is LocationGlobal:
        targetlocation=LocationGlobal(newlat, newlon,original_location.alt)
    elif type(original_location) is LocationGlobalRelative:
        targetlocation=LocationGlobalRelative(newlat, newlon,original_location.alt)
    else:
        raise Exception("Invalid Location object passed")
        
    return targetlocation;

def get_distance_metres(aLocation1, aLocation2):
    """
    Returns the ground distance in metres between two LocationGlobal objects.

    This method is an approximation, and will not be accurate over large distances and close to the 
    earth's poles. It comes from the ArduPilot test code: 
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5

def get_bearing(aLocation1, aLocation2):
    """
    Returns the bearing between the two LocationGlobal objects passed as parameters.

    This method is an approximation, and may not be accurate over large distances and close to the 
    earth's poles. It comes from the ArduPilot test code: 
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """	
    off_x = aLocation2.lon - aLocation1.lon
    off_y = aLocation2.lat - aLocation1.lat
    bearing = 90.00 + math.atan2(-off_y, off_x) * 57.2957795
    if bearing < 0:
        bearing += 360.00
    return bearing;

def goto_position_target_global_int(aLocation):
    """
    Send SET_POSITION_TARGET_GLOBAL_INT command to request the vehicle fly to a specified LocationGlobal.

    For more information see: https://pixhawk.ethz.ch/mavlink/#SET_POSITION_TARGET_GLOBAL_INT

    See the above link for information on the type_mask (0=enable, 1=ignore). 
    At time of writing, acceleration and yaw bits are ignored.
    """
    msg = vehicle.message_factory.set_position_target_global_int_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, # frame
        0b0000111111111000, # type_mask (only speeds enabled)
        aLocation.lat*1e7, # lat_int - X Position in WGS84 frame in 1e7 * meters
        aLocation.lon*1e7, # lon_int - Y Position in WGS84 frame in 1e7 * meters
        aLocation.alt, # alt - Altitude in meters in AMSL altitude, not WGS84 if absolute or relative, above terrain if GLOBAL_TERRAIN_ALT_INT
        0, # X velocity in NED frame in m/s
        0, # Y velocity in NED frame in m/s
        0, # Z velocity in NED frame in m/s
        0, 0, 0, # afx, afy, afz acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink) 
    # send command to vehicle
    vehicle.send_mavlink(msg)

def goto_position_target_local_ned(north, east, down):
    """	
    Send SET_POSITION_TARGET_LOCAL_NED command to request the vehicle fly to a specified 
    location in the North, East, Down frame.

    It is important to remember that in this frame, positive altitudes are entered as negative 
    "Down" values. So if down is "10", this will be 10 metres below the home altitude.

    Starting from AC3.3 the method respects the frame setting. Prior to that the frame was
    ignored. For more information see: 
    http://dev.ardupilot.com/wiki/copter-commands-in-guided-mode/#set_position_target_local_ned

    See the above link for information on the type_mask (0=enable, 1=ignore). 
    At time of writing, acceleration and yaw bits are ignored.

    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, # frame
        0b0000111111111000, # type_mask (only positions enabled)
        north, east, down, # x, y, z positions (or North, East, Down in the MAV_FRAME_BODY_NED frame
        0, 0, 0, # x, y, z velocity in m/s  (not used)
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink) 
    # send command to vehicle
    vehicle.send_mavlink(msg)

def goto(dNorth, dEast, gotoFunction=vehicle.simple_goto):
    """
    Moves the vehicle to a position dNorth metres North and dEast metres East of the current position.

    The method takes a function pointer argument with a single `dronekit.lib.LocationGlobal` parameter for 
    the target position. This allows it to be called with different position-setting commands. 
    By default it uses the standard method: dronekit.lib.Vehicle.simple_goto().

    The method reports the distance to target every two seconds.
    """
    
    currentLocation = vehicle.location.global_relative_frame
    targetLocation = get_location_metres(currentLocation, dNorth, dEast)
    targetDistance = get_distance_metres(currentLocation, targetLocation)
    gotoFunction(targetLocation)
    
    #print "DEBUG: targetLocation: %s" % targetLocation
    #print "DEBUG: targetLocation: %s" % targetDistance

    while vehicle.mode.name=="GUIDED": #Stop action if we are no longer in guided mode.
        #print "DEBUG: mode: %s" % vehicle.mode.name
        remainingDistance=get_distance_metres(vehicle.location.global_relative_frame, targetLocation)
        print("\n")
        print("Distance to target: ", remainingDistance)
        print("\n")
        if remainingDistance<=targetDistance*0.01: #Just below target, in case of undershoot.
            print("\n")
            print("Reached target")
            print("\n")
            break;
        time.sleep(2)

def send_ned_velocity(velocity_x, velocity_y, velocity_z, duration):
    """
    Move vehicle in direction based on specified velocity vectors and
    for the specified duration.

    This uses the SET_POSITION_TARGET_LOCAL_NED command with a type mask enabling only 
    velocity components 
    (http://dev.ardupilot.com/wiki/copter-commands-in-guided-mode/#set_position_target_local_ned).
    
    Note that from AC3.3 the message should be re-sent every second (after about 3 seconds
    with no message the velocity will drop back to zero). In AC3.2.1 and earlier the specified
    velocity persists until it is canceled. The code below should work on either version 
    (sending the message multiple times does not cause problems).
    
    See the above link for information on the type_mask (0=enable, 1=ignore). 
    At time of writing, acceleration and yaw bits are ignored.
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, # frame
        0b0000111111000111, # type_mask (only speeds enabled)
        0, 0, 0, # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z, # x, y, z velocity in m/s
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink) 

    # send command to vehicle on 1 Hz cycle
    for x in range(0,duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)
    
def send_global_velocity(velocity_x, velocity_y, velocity_z, duration):
    """
    Move vehicle in direction based on specified velocity vectors.

    This uses the SET_POSITION_TARGET_GLOBAL_INT command with type mask enabling only 
    velocity components 
    (http://dev.ardupilot.com/wiki/copter-commands-in-guided-mode/#set_position_target_global_int).
    
    Note that from AC3.3 the message should be re-sent every second (after about 3 seconds
    with no message the velocity will drop back to zero). In AC3.2.1 and earlier the specified
    velocity persists until it is canceled. The code below should work on either version 
    (sending the message multiple times does not cause problems).
    
    See the above link for information on the type_mask (0=enable, 1=ignore). 
    At time of writing, acceleration and yaw bits are ignored.
    """
    msg = vehicle.message_factory.set_position_target_global_int_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, # frame
        0b0000111111000111, # type_mask (only speeds enabled)
        0, # lat_int - X Position in WGS84 frame in 1e7 * meters
        0, # lon_int - Y Position in WGS84 frame in 1e7 * meters
        0, # alt - Altitude in meters in AMSL altitude(not WGS84 if absolute or relative)
        # altitude above terrain if GLOBAL_TERRAIN_ALT_INT
        velocity_x, # X velocity in NED frame in m/s
        velocity_y, # Y velocity in NED frame in m/s
        velocity_z, # Z velocity in NED frame in m/s
        0, 0, 0, # afx, afy, afz acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink) 

    # send command to vehicle on 1 Hz cycle
    for x in range(0,duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)

def distance_to_current_waypoint():
    """
    Gets distance in metres to the current waypoint. 
    It returns None for the first waypoint (Home location).
    """
    nextwaypoint = vehicle.commands.next
    if nextwaypoint==0:
        return None
    missionitem=vehicle.commands[nextwaypoint-1] #commands are zero indexed
    lat = missionitem.x
    lon = missionitem.y
    alt = missionitem.z
    targetWaypointLocation = LocationGlobalRelative(lat,lon,alt)
    distancetopoint = get_distance_metres(vehicle.location.global_frame, targetWaypointLocation)
    return distancetopoint

def download_mission():
    """
    Download the current mission from the vehicle.
    """
    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready() # wait until download is complete.

def adds_square_mission(aLocation, aSize):
    """
    Adds a takeoff command and four waypoint commands to the current mission. 
    The waypoints are positioned to form a square of side length 2*aSize around the specified LocationGlobal (aLocation).

    The function assumes vehicle.commands matches the vehicle mission state 
    (you must have called download at least once in the session and after clearing the mission)
    """	

    cmds = vehicle.commands
    
    print("\n")
    print("[INFO] Clear any existing commands")
    print("\n")
    cmds.clear() 
    
    print("\n")
    print("[INFO] Define/add new commands.")
    print("\n")
    # Add new commands. The meaning/order of the parameters is documented in the Command class. 
     
    #Add MAV_CMD_NAV_TAKEOFF command. This is ignored if the vehicle is already in the air.
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 0, 10))

    #Define the four MAV_CMD_NAV_WAYPOINT locations and add the commands
    point1 = get_location_metres(aLocation, aSize, -aSize)
    point2 = get_location_metres(aLocation, aSize, aSize)
    point3 = get_location_metres(aLocation, -aSize, aSize)
    point4 = get_location_metres(aLocation, -aSize, -aSize)
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, 0, 0, point1.lat, point1.lon, 11))
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, 0, 0, point2.lat, point2.lon, 12))
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, 0, 0, point3.lat, point3.lon, 13))
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, 0, 0, point4.lat, point4.lon, 14))
    #add dummy waypoint "5" at point 4 (lets us know when have reached destination)
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, 0, 0, point4.lat, point4.lon, 14))    
    
    print("\n")
    print("[INFO] Upload new commands to vehicle")
    print("\n")
    cmds.upload()

# example for Simple GoTo mission
"""
Fly a triangular path using the standard Vehicle.simple_goto() method.

The method is called indirectly via a custom "goto" that allows the target position to be
specified as a distance in metres (North/East) from the current position, and which reports
the distance-to-target.
"""	
# print("TRIANGLE path using standard Vehicle.simple_goto()")

# print("Set groundspeed to 5m/s.")
# vehicle.groundspeed=5

# print("Position North 80 West 50")
# goto(80, -50)

# print("Position North 0 East 100")
# goto(0, 100)

# print("Position North -80 West 50")
# goto(-80, -50)

# example for Simple GoTo mission in global positioning
"""
Fly a triangular path using the SET_POSITION_TARGET_GLOBAL_INT command and specifying
a target position (rather than controlling movement using velocity vectors). The command is
called from goto_position_target_global_int() (via `goto`).

The goto_position_target_global_int method is called indirectly from a custom "goto" that allows 
the target position to be specified as a distance in metres (North/East) from the current position, 
and which reports the distance-to-target.

The code also sets the speed (MAV_CMD_DO_CHANGE_SPEED). In AC3.2.1 Copter will accelerate to this speed 
near the centre of its journey and then decelerate as it reaches the target. 
In AC3.3 the speed changes immediately.
"""	
# print("TRIANGLE path using standard SET_POSITION_TARGET_GLOBAL_INT message and with varying speed.")
# print("Position South 100 West 130")

# print("Set groundspeed to 5m/s.")
# vehicle.groundspeed = 5
# goto(-100, -130, goto_position_target_global_int)

# print("Set groundspeed to 15m/s (max).")
# vehicle.groundspeed = 15
# print("Position South 0 East 200")
# goto(0, 260, goto_position_target_global_int)

# print("Set airspeed to 10m/s (max).")
# vehicle.airspeed = 10

# print("Position North 100 West 130")
# goto(100, -130, goto_position_target_global_int)

# Example for Simple GoTo mission in local positioning
"""
Fly the vehicle in a 50m square path, using the SET_POSITION_TARGET_LOCAL_NED command 
and specifying a target position (rather than controlling movement using velocity vectors). 
The command is called from goto_position_target_local_ned() (via `goto`).

The position is specified in terms of the NED (North East Down) relative to the Home location.

WARNING: The "D" in NED means "Down". Using a positive D value will drive the vehicle into the ground!

The code sleeps for a time (DURATION) to give the vehicle time to reach each position (rather than 
sending commands based on proximity).

The code also sets the region of interest (MAV_CMD_DO_SET_ROI) via the `set_roi()` method. This points the 
camera gimbal at the the selected location (in this case it aligns the whole vehicle to point at the ROI).
"""	

# print("SQUARE path using SET_POSITION_TARGET_LOCAL_NED and position parameters")
# DURATION = 20 #Set duration for each segment.

# print("North 50m, East 0m, 10m altitude for %s seconds" % DURATION)
# goto_position_target_local_ned(50,0,-10)
# print("Point ROI at current location (home position)") 
# # NOTE that this has to be called after the goto command as first "move" command of a particular type
# # "resets" ROI/YAW commands
# set_roi(vehicle.location.global_relative_frame)
# time.sleep(DURATION)

# print("North 50m, East 50m, 10m altitude")
# goto_position_target_local_ned(50,50,-10)
# time.sleep(DURATION)

# print("Point ROI at current location")
# set_roi(vehicle.location.global_relative_frame)

# print("North 0m, East 50m, 10m altitude")
# goto_position_target_local_ned(0,50,-10)
# time.sleep(DURATION)

# print("North 0m, East 0m, 10m altitude")
# goto_position_target_local_ned(0,0,-10)
# time.sleep(DURATION)

# Example for Simple GoTo mission in square pattern
"""
Fly the vehicle in a SQUARE path using velocity vectors (the underlying code calls the 
SET_POSITION_TARGET_LOCAL_NED command with the velocity parameters enabled).

The thread sleeps for a time (DURATION) which defines the distance that will be travelled.

The code also sets the yaw (MAV_CMD_CONDITION_YAW) using the `set_yaw()` method in each segment
so that the front of the vehicle points in the direction of travel
"""


#Set up velocity vector to map to each direction.
# # vx > 0 => fly North
# # vx < 0 => fly South
# NORTH = 2
# SOUTH = -2

# # Note for vy:
# # vy > 0 => fly East
# # vy < 0 => fly West
# EAST = 2
# WEST = -2

# # Note for vz: 
# # vz < 0 => ascend
# # vz > 0 => descend
# UP = -0.5
# DOWN = 0.5

# DURATION = 20 #Set duration for each segment.


# # Square path using velocity
# print("SQUARE path using SET_POSITION_TARGET_LOCAL_NED and velocity parameters")

# print("Yaw 180 absolute (South)")
# condition_yaw(180)

# print("Velocity South & up")
# send_ned_velocity(SOUTH,0,UP,DURATION)
# send_ned_velocity(0,0,0,1)


# print("Yaw 270 absolute (West)")
# condition_yaw(270)

# print("Velocity West & down")
# send_ned_velocity(0,WEST,DOWN,DURATION)
# send_ned_velocity(0,0,0,1)


# print("Yaw 0 absolute (North)")
# condition_yaw(0)

# print("Velocity North")
# send_ned_velocity(NORTH,0,0,DURATION)
# send_ned_velocity(0,0,0,1)


# print("Yaw 90 absolute (East)")
# condition_yaw(90)

# print("Velocity East")
# send_ned_velocity(0,EAST,0,DURATION)
# send_ned_velocity(0,0,0,1)


# Example for Simple GoTo mission in diamond pattern
"""
Fly the vehicle in a DIAMOND path using velocity vectors (the underlying code calls the 
SET_POSITION_TARGET_GLOBAL_INT command with the velocity parameters enabled).

The thread sleeps for a time (DURATION) which defines the distance that will be travelled.

The code sets the yaw (MAV_CMD_CONDITION_YAW) using the `set_yaw()` method using relative headings
so that the front of the vehicle points in the direction of travel.

At the end of the second segment the code sets a new home location to the current point.
"""
#Set up velocity vector to map to each direction.
# # vx > 0 => fly North
# # vx < 0 => fly South
# NORTH = 2
# SOUTH = -2

# # Note for vy:
# # vy > 0 => fly East
# # vy < 0 => fly West
# EAST = 2
# WEST = -2

# # Note for vz: 
# # vz < 0 => ascend
# # vz > 0 => descend
# UP = -0.5
# DOWN = 0.5

# DURATION = 20 #Set duration for each segment.

# print("DIAMOND path using SET_POSITION_TARGET_GLOBAL_INT and velocity parameters")
# # vx, vy are parallel to North and East (independent of the vehicle orientation)

# print("[INFO] Yaw 225 absolute")
# condition_yaw(225)

# print("[INFO] Velocity South, West and Up")
# send_global_velocity(SOUTH,WEST,UP,DURATION)
# send_global_velocity(0,0,0,1)


# print("[INFO] Yaw 90 relative (to previous yaw heading)")
# condition_yaw(90,relative=True)

# print("[INFO] Velocity North, West and Down")
# send_global_velocity(NORTH,WEST,DOWN,DURATION)
# send_global_velocity(0,0,0,1)

# print("[INFO] Set new home location to current location")
# vehicle.home_location=vehicle.location.global_frame
# print("[INFO] Get new home location")
# #This reloads the home location in DroneKit and GCSs
# cmds = vehicle.commands
# cmds.download()
# cmds.wait_ready()
# print("[INFO] Home Location: %s" % vehicle.home_location)


# print("[INFO] Yaw 90 relative (to previous yaw heading)")
# condition_yaw(90,relative=True)

# print("[INFO] Velocity North and East")
# send_global_velocity(NORTH,EAST,0,DURATION)
# send_global_velocity(0,0,0,1)

# print("[INFO] Yaw 90 relative (to previous yaw heading)")
# condition_yaw(90,relative=True)

# print("[INFO] Velocity South and East")
# send_global_velocity(SOUTH,EAST,0,DURATION)
# send_global_velocity(0,0,0,1)

# Example to land the vehicle
# print("\n")
# print("[INFO] Setting LAND mode...")
# print("\n")
# vehicle.mode = VehicleMode("LAND")

# #Close vehicle object before exiting script
# print("\n")
# print("[INFO] Close vehicle object")
# print("\n")
# vehicle.close()

def land_mission():
  # activate only for indoor mission only
  vehicle.mode = VehicleMode("LOITER")
  print("\n")
  print("[INFO] Vehicle in hold position...")
  print("\n")

  vehicle.mode = VehicleMode("LAND")
  print("\n")
  print("[INFO] Vehicle is landing...")
  print("\n")

  disarm_vehicle()

def disarm_vehicle():
    # disarming the vehicle
    vehicle.disarm = True
    print ("\n[INFO] Set Vehicle.disarm=True (currently: %s)" % vehicle.disarm)
    while not vehicle.disarm:
        print("\n")
        print ("[INFO] Waiting for disarming...")
        time.sleep(1)
        print ("[INFO] Vehicle is disarmed: %s" % vehicle.disarm)
        print("\n")

    # close vehicle object before exiting script
    vehicle.close()
    print("\n")
    print("[INFO] Close vehicle object")
    print("\n")

def battery_level():
  print("\n")
  print("{INFO] Battery Level: %s" % vehicle.battery)
  print("\n")

def vehicle_altitude():
  print("\n")
  print("[INFO] Altitude : {0.2f} m".format(vehicle.location.global_relative_frame.alt)) 
  print("\n")

def emergency_land():
    vehicle.mode = VehicleMode("LAND")
    print("[INFO] Vehicle is landing...")
    print("\n")

def forward_payload_mission():
    # # activate only for indoor nmission only
    # vehicle.mode = VehicleMode("LOITER")
    # current_roll = vehicle.attitude.roll
    # current_heading = vehicle.heading
    # target_heading = start_heading
    # if distance_left < distance_left_minimum and distance_right < distance_right_minimum:
    #   print("\n")
    #   print("[INFO] DANGEROUS DISTANCE TO SIDE OBJECTS... !!!")
    #   print("\n")
    #   if current_heading < target_heading:
    #     deviation_heading = target_heading - current_heading
    #     vehicle.mode = VehicleMode("LOITER")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(yaw_angle=(current_heading + deviation_heading))
    #     vehicle.mode = VehicleMode("LOITER")
    #   elif current_heading > target_heading:      
    #     deviation_heading = current_heading - target_heading
    #     vehicle.mode = VehicleMode("LOITER")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(yaw_angle=(current_heading - deviation_heading))
    #     vehicle.mode = VehicleMode("LOITER")
    #   if current_roll < target_roll:      
    #     deviation_roll = target_roll - current_roll
    #     vehicle.mode = VehicleMode("LOITER")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(roll_angle=(current_roll + deviation_roll))
    #     vehicle.mode = VehicleMode("LOITER")
    #   elif current_roll > target_roll:
    #     deviation_roll = current_roll - target_roll
    #     vehicle.mode = VehicleMode("LOITER")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(roll_angle=(current_roll - deviation_roll))
    #     vehicle.mode = VehicleMode("LOITER")
    
    # if distance_left == distance_left_minimum and distance_right == distance_right_minimum:
    #   print("\n")
    #   print("[INFO] Safe Distance to Side Objects... !!!")
    #   print("\n")
    #   # activate only for the indoor mission only  
    #   # Move the drone forward
    #   print("\n")
    #   print("[INFO] Moving forward to payload position")
    #   print("\n")

    #   vehicle.mode = VehicleMode("AUTO")

    #   vehicle.commands.next = payload_waypoint # going to payload location
    #   print("\n")
    #   print("[INFO] Current Waypoint: ", vehicle.commands.next)
    #   print("\n")
    #   if vehicle.commands.next == payload_waypoint():
    #    print("\n")
    #    print("[INFO] Vehicle in payload_waypoint...")
    #    print("\n")

    #   vehicle.commands.next = payload_loiter_waypoint # going to payload location
    #   print("\n")
    #   print("[INFO] Current Waypoint: ", vehicle.commands.next)
    #   print("\n")
    #   if vehicle.commands.next == payload_loiter_waypoint():
    #    print("\n")
    #    print("[INFO] Vehicle in payload_loiter_waypoint...")
    #    print("\n")
  
    #    print("\n")
    #    print("[INFO] Hold position and searching the payload...")
    #    print("\n")

    # activate only for indoor and outdoor mission combination
    #  moving forward to the payload position
    vehicle.mode = VehicleMode("LOITER")
    vehicle.mode = VehicleMode("GUIDED")
    current_roll = vehicle.attitude.roll
    current_heading = vehicle.heading
    target_heading = start_heading
    if current_heading < target_heading:
       deviation_heading = target_heading - current_heading
       send_attitude_target(yaw_angle=(current_heading + deviation_heading), thrust=0.5)
       vehicle.mode = VehicleMode("LOITER")
    elif current_heading > target_heading:
       deviation_heading = current_heading - target_heading
       send_attitude_target(yaw_angle=(current_heading - deviation_heading), thrust=0.5)
       vehicle.mode = VehicleMode("LOITER")
    if current_roll < target_roll:
       deviation_roll = target_roll - current_roll
       send_attitude_target(roll_angle=(current_roll + deviation_roll), thrust=0.5)
       vehicle.mode = VehicleMode("LOITER")
    elif current_roll > target_roll:
       deviation_roll = current_roll - target_roll
       send_attitude_target(roll_angle=(current_roll - deviation_roll), thrust=0.5)
       vehicle.mode = VehicleMode("LOITER")
       print("\n")
       print("[INFO] Moving forward to corner position")
       print("\n")
       send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_payload)
       send_attitude_target(pitch_angle = +5, thrust = 0.5)
       vehicle.mode = VehicleMode("LOITER")
       print("\n")
       print("[INFO] Hold position and searching the payload...")
       print("\n")

def forward_corner_misssion():
    # # activate only for indoor nmission only   
    # vehicle.mode = VehicleMode("LOITER")
    # current_roll = vehicle.attitude.roll
    # current_heading = vehicle.heading
    # target_heading = start_heading
    # if distance_left < distance_left_minimum and distance_right < distance_right_minimum:      
    #   print("\n")
    #   print("[INFO] DANGEROUS DISTANCE TO SODE OBJECTS... !!!")
    #   print("\n")
    #   if current_heading < target_heading:
    #     deviation_heading = target_heading - current_heading
    #     vehicle.mode = VehicleMode("LOITER")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(yaw_angle=(current_heading + deviation_heading))
    #     vehicle.mode = VehicleMode("LOITER")
    #   elif current_heading > target_heading:      
    #     deviation_heading = current_heading - target_heading
    #     vehicle.mode = VehicleMode("LOITER")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(yaw_angle=(current_heading - deviation_heading))
    #     vehicle.mode = VehicleMode("LOITER")
    #   if current_roll < target_roll:      
    #     deviation_roll = target_roll - current_roll
    #     vehicle.mode = VehicleMode("LOITER")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(roll_angle=(current_roll + deviation_roll))
    #     vehicle.mode = VehicleMode("LOITER")
    #   elif current_roll > target_roll:
    #     deviation_roll = current_roll - target_roll
    #     vehicle.mode = VehicleMode("LOITER")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(roll_angle=(current_roll - deviation_roll))
    #     vehicle.mode = VehicleMode("LOITER")
    
    # if distance_left == distance_left_minimum and distance_right == distance_right_minimum:
    #   print("\n")
    #   print("[INFO] Safe Distance to Side Objects... !!!")
    #   print("\n")
    #   # Move the drone forward
    #   # activate only for indoor mission only
    #   print("\n")
    #   print("[INFO] Moving forward to corner position")
    #   print("\n")

    #   vehicle.mode = VehicleMode("AUTO")
    #   vehicle.commands.next = forward_corner_waypoint # going to corner location
    #   print("\n")
    #   print("[INFO] Current Waypoint: ", vehicle.commands.next)
    #   print("\n")
    #   if vehicle.commands.next == forward_corner_waypoint():
    #    print("\n")
    #    print("[INFO] Vehicle in forward_corner_waypoint...")
    #    print("\n")
      

    #   vehicle.commands.next = corner_loiter_waypoint() # going to corner location and hold position
    #   print("\n")
    #   print("[INFO] Current Waypoint: ", vehicle.commands.next)
    #   print("\n")
    #   if vehicle.commands.next == corner_loiter_waypoint():
    #    print("\n")
    #    print("[INFO] Vehicle in corner_loiter_waypoint...")
    #    print("\n")

    #   if distance_front == distance_front_minimum and distance_left == distance_left_minimum and distance_right == distance_right_minimum:
    #     vehicle.mode = VehicleMode("LOITER")  
    #     print("\n")
    #     print("[INFO] Vehicle hold in corner position...")
    #     print("\n")

    # activate only for indoor and outdoor mission combination
    # moving forward to the corner position
    vehicle.mode = VehicleMode("LOITER")
    vehicle.mode = VehicleMode("GUIDED")
    current_roll = vehicle.attitude.roll
    current_heading = vehicle.heading
    target_heading = start_heading
    if current_heading < target_heading:
      deviation_heading = target_heading - current_heading
      send_attitude_target(yaw_angle=(current_heading + deviation_heading), thrust=0.5)
      vehicle.mode = VehicleMode("LOITER")
    elif current_heading > target_heading:
      deviation_heading = current_heading - target_heading
      send_attitude_target(yaw_angle=(current_heading - deviation_heading), thrust=0.5)
      vehicle.mode = VehicleMode("LOITER")
    if current_roll < target_roll:
      deviation_roll = target_roll - current_roll
      send_attitude_target(roll_angle=(current_roll + deviation_roll), thrust=0.5)
      vehicle.mode = VehicleMode("LOITER")
    elif current_roll > target_roll:
      deviation_roll = current_roll - target_roll
      send_attitude_target(roll_angle=(current_roll - deviation_roll), thrust=0.5)
      vehicle.mode = VehicleMode("LOITER")
      print("\n")
      print("[INFO] Moving forward to corner position")
      print("\n")
      send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_box_1)
      send_attitude_target(pitch_angle = +5, thrust = 0.5)

    if distance_front == distance_front_minimum:
      vehicle.mode = VehicleMode("LOITER")
      print("\n")
      print("[INFO] Vehicle hold in corner position...")
      print("\n")

    if distance_front < distance_front_minimum:
      vehicle.mode = VehicleMode("LAND")
      print("\n")
      print("[INFO] Vehicle is landing...")
      print("\n")
      disarm_vehicle()

def moving_left_right_mission():
    # # activate only for indoor mission only
    # vehicle.commands.next = corner_loiter_waypoint() # going to corner location and hold position
    # print("\n")
    # print("[INFO] Current Waypoint: ", vehicle.commands.next)
    # print("\n")
    # if vehicle.commands.next == corner_loiter_waypoint():
    #    print("\n")
    #    print("[INFO] Vehicle in corner_loiter_waypoint...")
    #    print("\n")
    # current_roll = vehicle.attitude.roll
    # current_heading = vehicle.heading
    # if yaw_angle_option == 1:
    #   target_heading = yaw_angle_right
    # if yaw_angle_option == -1:
    #   target_heading = yaw_angle_left    
    # if distance_left < distance_left_minimum and distance_right < distance_right_minimum:
    #   print("\n")
    #   print("[INFO] DANGEROUS DISTANCE TO SIDE OBJECTS... !!!")
    #   print("\n")
    #   if current_heading < target_heading:
    #     deviation_heading = target_heading - current_heading
    #     vehicle.mode = VehicleMode("LOITER")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(yaw_angle=(current_heading + deviation_heading))
    #     vehicle.mode = VehicleMode("LOITER")
    #   elif current_heading > target_heading:      
    #     deviation_heading = current_heading - target_heading
    #     vehicle.mode = VehicleMode("LOITER")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(yaw_angle=(current_heading - deviation_heading))
    #     vehicle.mode = VehicleMode("LOITER")
    #   if current_roll < target_roll:      
    #     deviation_roll = target_roll - current_roll
    #     vehicle.mode = VehicleMode("LOITER")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(roll_angle=(current_roll + deviation_roll))
    #     vehicle.mode = VehicleMode("LOITER")
    #   elif current_roll > target_roll:
    #     deviation_roll = current_roll - target_roll
    #     vehicle.mode = VehicleMode("LOITER")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(roll_angle=(current_roll - deviation_roll))
    #     vehicle.mode = VehicleMode("LOITER")

    # vehicle.mode = VehicleMode("LOITER")
    # print("\n")
    # print("[INFO] Hold position in corner position...")
    # print("\n")

    # if distance_left == distance_left_minimum and distance_right == distance_right_minimum:
    #   print("\n")
    #   print("[INFO] Safe Distance to Side Objects... !!!")
    #   print("\n")
    #   # activate only for indoor mission only
    #   # moving forward to the corner position
    #   vehicle.mode = VehicleMode("AUTO")
    #   print("\n")
    #   print("[INFO] Vehicle is going forward to drop location")
    #   print("\n")

    #   vehicle.commands.next = corner_forward_waypoint() # going forward to drop location
    #   print("\n")
    #   print("[INFO] Current Waypoint: ", vehicle.commands.next)
    #   print("\n")
    #   if vehicle.commands.next == corner_forward_waypoint():
    #    print("\n")
    #    print("[INFO] Vehicle in corner_forward_waypoint...")
    #    print("\n")

    #   vehicle.commands.next = drop_waypoint() # going forward to drop location
    #   print("\n")
    #   print("[INFO] Current Waypoint: ", vehicle.commands.next)
    #   print("\n")
    #   if vehicle.commands.next == drop_waypoint():
    #    print("\n")
    #    print("[INFO] Vehicle in drop_waypoint...")
    #    print("\n")

    #   vehicle.commands.next = drop_loiter_waypoint() # going forward to drop location
    #   print("\n")
    #   print("[INFO] Current Waypoint: ", vehicle.commands.next)
    #   print("\n")
    #   if vehicle.commands.next == drop_loiter_waypoint():
    #     print("\n")
    #     print("[INFO] Vehicle in drop_loiter_waypoint...")
    #     print("\n")

    #     print("\n")
    #     print("[INFO] Hold position in payload drop position...")
    #     print("\n")

    #     vehicle.mode = VehicleMode("LOITER")
    #     print("\n")
    #     print("[INFO] Hold position in payload drop position...")
    #     print("\n")

    # activate only for indoor and outdoor mission combination
    # vehicle moving left or right (kotak kiri atau kanan)
    vehicle.mode = VehicleMode("LOITER")
    print("\n")
    print("[INFO] Hold position in corner position...")
    print("\n")
    vehicle.mode = VehicleMode("GUIDED")
    current_roll = vehicle.attitude.roll
    current_heading = vehicle.heading
    if yaw_angle_option == -1:
      send_attitude_target(pitch_angle=0, yaw_angle = yaw_angle_left, thrust = 0.5)
      if current_roll < target_roll:
        deviation_roll = target_roll - current_roll
        send_attitude_target(roll_angle=(current_roll + deviation_roll), thrust=0.5)
      elif current_roll > target_roll:
        deviation_roll = current_roll - target_roll
        send_attitude_target(roll_angle=(current_roll - deviation_roll), thrust=0.5)
      print("\n")
      print("[INFO] Moving left to drop position")
      print("\n")
        
    if yaw_angle_option == 1:
      send_attitude_target(pitch_angle=0, yaw_angle = yaw_angle_right, thrust = 0.5)
      if current_roll < target_roll:
        deviation_roll = target_roll - current_roll
        send_attitude_target(roll_angle=(current_roll + deviation_roll), thrust=0.5)
      elif current_roll > target_roll:
        deviation_roll = current_roll - target_roll
        send_attitude_target(roll_angle=(current_roll - deviation_roll), thrust=0.5)
      print("\n")
      print("[INFO] Moving right to drop position")
      print("\n")
    
    # moving forward to the box position
    if vehicle.location.global_relative_frame.alt == altitude_drop:
        send_attitude_target(pitch_angle = 0, thrust = 0.5)
        send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_box_2)
    if vehicle.location.global_relative_frame.alt <= altitude_drop:
        send_attitude_target(pitch_angle = 0, thrust = 0.7)
        if vehicle.location.global_relative_frame.alt == altitude_drop:
          send_attitude_target(pitch_angle = 0, thrust = 0.5)    
          send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_box_2)
    if vehicle.location.global_relative_frame.alt >= altitude_drop:
        send_attitude_target(pitch_angle = 0, thrust = 0.3)
        if vehicle.location.global_relative_frame.alt == altitude_drop:
          send_attitude_target(pitch_angle = 0, thrust = 0.5)
          send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_box_2)
    if current_roll < target_roll:
        deviation_roll = target_roll - current_roll
        send_attitude_target(roll_angle=(current_roll + deviation_roll), thrust=0.5)
    elif current_roll > target_roll:
        deviation_roll = current_roll - target_roll
        send_attitude_target(roll_angle=(current_roll - deviation_roll), thrust=0.5)
    if current_heading < yaw_angle_right:
        deviation_heading = yaw_angle_right - current_heading
        send_attitude_target(yaw_angle=(current_heading + deviation_heading), thrust=0.5)
    elif current_heading > yaw_angle_right:
        deviation_heading = current_heading - yaw_angle_right
        send_attitude_target(yaw_angle=(current_heading - deviation_heading), thrust=0.5)

    print("\n")
    print("[INFO] Moving forward to drop position")
    print("\n")

    # Hold the position and searching the box.
    vehicle.mode = VehicleMode("LOITER")
    if current_roll < target_roll:
      vehicle.mode = VehicleMode("GUIDED")
      deviation_roll = target_roll - current_roll
      send_attitude_target(roll_angle=(current_roll + deviation_roll), thrust=0.5)
    elif current_roll > target_roll:
      vehicle.mode = VehicleMode("GUIDED")
      deviation_roll = target_roll - target_roll
      send_attitude_target(roll_angle=(current_roll - deviation_roll), thrust=0.5)
      vehicle.mode = VehicleMode("LOITER")
    if current_heading < yaw_angle_right:
      vehicle.mode = VehicleMode("GUIDED")
      deviation_heading = yaw_angle_right - current_heading
      send_attitude_target(yaw_angle=(current_heading + deviation_heading), thrust=0.5)
      vehicle.mode = VehicleMode("LOITER")
    elif current_heading > yaw_angle_right:
      vehicle.mode = VehicleMode("GUIDED")
      deviation_heading = current_heading - yaw_angle_right
      send_attitude_target(yaw_angle=(current_heading - deviation_heading), thrust=0.5)
      vehicle.mode = VehicleMode("LOITER")
    print("\n")
    print("[INFO] Hold position and searching the box...")
    print("\n")

def drop_payload_mission():
    # activate onyl for indoor mission, also only indoor and outdoot mission combination
    vehicle.mode = VehicleMode("LOITER")
    print("\n")
    print("[INFO] Hold position in payload drop position...")
    print("\n")
    current_roll = vehicle.attitude.roll
    current_heading = vehicle.heading
    if yaw_angle_option == 1:
      target_heading = yaw_angle_right
    if yaw_angle_option == -1:
      target_heading = yaw_angle_left
    if distance_left < distance_left_minimum and distance_right < distance_right_minimum:
      print("\n")
      print("[INFO] DANGEROUS DISTANCE TO SIDE OBJECTS... !!!")
      print("\n")
      if current_heading < target_heading:
        deviation_heading = target_heading - current_heading
        vehicle.mode = VehicleMode("LOITER")
        vehicle.mode = VehicleMode("GUIDED")
        send_attitude_target(yaw_angle=(current_heading + deviation_heading))
        vehicle.mode = VehicleMode("LOITER")
      elif current_heading > target_heading:      
        deviation_heading = current_heading - target_heading
        vehicle.mode = VehicleMode("LOITER")
        vehicle.mode = VehicleMode("GUIDED")
        send_attitude_target(yaw_angle=(current_heading - deviation_heading))
        vehicle.mode = VehicleMode("LOITER")
      if current_roll < target_roll:      
        deviation_roll = target_roll - current_roll
        vehicle.mode = VehicleMode("LOITER")
        vehicle.mode = VehicleMode("GUIDED")
        send_attitude_target(roll_angle=(current_roll + deviation_roll))
        vehicle.mode = VehicleMode("LOITER")
      elif current_roll > target_roll:
        deviation_roll = current_roll - target_roll
        vehicle.mode = VehicleMode("LOITER")
        vehicle.mode = VehicleMode("GUIDED")
        send_attitude_target(roll_angle=(current_roll - deviation_roll))
        vehicle.mode = VehicleMode("LOITER")

    if distance_left == distance_left_minimum and distance_right == distance_right_minimum and vehicle.commands.next == drop_loiter_waypoint():
      print("\n")
      print("[INFO] Safe Distance to Side Objects... !!!")
      print("\n")
      
      print("\n")
      print("[INFO] Vehicle in drop_loiter_waypoint...")
      print("\n")
      
      vehicle.mode = VehicleMode("LOITER")
      print("\n")
      print("[INFO] Hold position in payload drop position...")
      print("\n")

def window_pass_mission_before():
    # # activate only for indoor mission only
    # vehicle.mode = VehicleMode("LOITER")
    # print("\n")
    # print("[INFO] Hold position in payload drop position...")
    # print("\n")
    # current_roll = vehicle.attitude.roll
    # current_heading = vehicle.heading
    # if yaw_angle_option == 1:
    #   target_heading = yaw_angle_right
    # if yaw_angle_option == -1:
    #   target_heading = yaw_angle_left
    # if distance_left < distance_left_minimum and distance_right < distance_right_minimum:
    #   print("\n")
    #   print("[INFO] DANGEROUS DISTANCE TO SIDE OBJECTS... !!!")
    #   print("\n")
    #   if current_heading < target_heading:
    #     deviation_heading = target_heading - current_heading
    #     vehicle.mode = VehicleMode("LOITER")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(yaw_angle=(current_heading + deviation_heading))
    #     vehicle.mode = VehicleMode("LOITER")
    #   elif current_heading > target_heading:      
    #     deviation_heading = current_heading - target_heading
    #     vehicle.mode = VehicleMode("LOITER")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(yaw_angle=(current_heading - deviation_heading))
    #     vehicle.mode = VehicleMode("LOITER")
    #   if current_roll < target_roll:      
    #     deviation_roll = target_roll - current_roll
    #     vehicle.mode = VehicleMode("LOITER")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(roll_angle=(current_roll + deviation_roll))
    #     vehicle.mode = VehicleMode("LOITER")
    #   elif current_roll > target_roll:
    #     deviation_roll = current_roll - target_roll
    #     vehicle.mode = VehicleMode("LOITER")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(roll_angle=(current_roll - deviation_roll))
    #     vehicle.mode = VehicleMode("LOITER")

    # if distance_left == distance_left_minimum and distance_right == distance_right_minimum and vehicle.commands.next == drop_loiter_waypoint():
    #   print("\n")
    #   print("[INFO] Safe Distance to Side Objects... !!!")
    #   print("\n")
    #   # activate only for indoor mission only
    #   vehicle.mode = VehicleMode("LOITER")
    #   print("\n")
    #   print("[INFO] Hold position in payload drop position...")
    #   print("\n")

    #   vehicle.mode = VehicleMode("AUTO")

    #   print("\n")
    #   print("[INFO] Going to window position...")
    #   print("\n")

    #   vehicle.commands.next = window_loiter_waypoint # going forward to window location
    #   print("\n")
    #   print("[INFO] Current Waypoint: ", vehicle.commands.next)
    #   print("\n")
    #   if vehicle.commands.next == window_loiter_waypoint():
    #     print("\n")
    #     print("[INFO] Vehicle in window_loiter_waypoint...")
    #     print("\n")
  
    #   vehicle.commands.next = window_forward_waypoint_before_indoor # going forward to window location
    #   print("\n")
    #   print("[INFO] Current Waypoint: ", vehicle.commands.next)
    #   print("\n")
    #   if vehicle.commands.next == window_forward_waypoint_before_indoor():
    #     print("\n")
    #     print("[INFO] Vehicle in window_forward_waypoint_before_indoor...")
    #     print("\n")
    
    # if distance_front > distance_front_minimum:
    #     vehicle.mode = VehicleMode("AUTO")
    #     vehicle.commands.next = window_forward_waypoint_before_outdoor # going forward to window location
    #     print("\n")
    #     print("[INFO] Current Waypoint: ", vehicle.commands.next)
    #     print("\n")
    #     if vehicle.commands.next == window_forward_waypoint_before_outdoor():
    #         print("\n")
    #         print("[INFO] Vehicle in window_forward_waypoint_before_outdoor...")
    #         print("\n")

    # if distance_front == distance_front_minimum:
    #     vehicle.mode = VehicleMode("LOITER")
    #     print("\n")
    #     print("[INFO] Vehicle in hold position...")
    #     print("\n")
           
    # activate only for indoor and outdoor mission combination
    # vehicle.mode = VehicleMode("LOITER")
    print("\n")
    print("[INFO] Vehicle in hold position...")
    print("\n")

    vehicle.mode = VehicleMode("GUIDED")
    current_roll = vehicle.attitude.roll
    current_heading = vehicle.heading
    if yaw_angle_option == -1:
      send_attitude_target(pitch_angle=0, yaw_angle = yaw_angle_left, thrust = 0.5)
      if current_roll < target_roll:
         deviation_roll = target_roll - current_roll
         send_attitude_target(roll_angle=(current_roll + deviation_roll), thrust=0.5)
      elif current_roll > target_roll:
         deviation_roll = current_roll - target_roll
         send_attitude_target(roll_angle=(current_roll - deviation_roll), thrust=0.5)
      print("\n")
      print("[INFO] Moving left to drop position")
      print("\n")
        
    if yaw_angle_option == 1:
      send_attitude_target(pitch_angle=0, yaw_angle = yaw_angle_right, thrust = 0.5)
      if current_roll < target_roll:
        deviation_roll = target_roll - current_roll
        send_attitude_target(roll_angle=(current_roll + deviation_roll), thrust=0.5)
      elif current_roll > target_roll:
        deviation_roll = current_roll - target_roll
        send_attitude_target(roll_angle=(current_roll - deviation_roll), thrust=0.5)
      print("\n")
      print("[INFO] Moving right to drop position")
      print("\n")
    # moving forward to the window position
    print("\n")
    print("[INFO] Moving forward to window position")
    print("\n")
    if vehicle.location.global_relative_frame.alt == altitude_drop:
        send_attitude_target(pitch_angle = 0, thrust = 0.5)
        send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_window)
    if vehicle.location.global_relative_frame.alt <= altitude_drop:
        send_attitude_target(pitch_angle = 0, thrust = 0.7)
        if vehicle.location.global_relative_frame.alt == altitude_drop:
          send_attitude_target(pitch_angle = 0, thrust = 0.5)    
          send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_window)
    if vehicle.location.global_relative_frame.alt >= altitude_drop:
        send_attitude_target(pitch_angle = 0, thrust = 0.3)
        if vehicle.location.global_relative_frame.alt == altitude_drop:
          send_attitude_target(pitch_angle = 0, thrust = 0.5)
          send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_window)
    if current_roll < target_roll:
      deviation_roll = target_roll - current_roll
      send_attitude_target(roll_angle=(current_roll + deviation_roll), thrust=0.5)
    elif current_roll > target_roll:
      deviation_roll = current_roll - target_roll
      send_attitude_target(roll_angle=(current_roll - deviation_roll), thrust=0.5)
    if current_heading < yaw_angle_right:
      deviation_heading = yaw_angle_right - current_heading
      send_attitude_target(yaw_angle=(current_heading + deviation_heading), thrust=0.5)
    elif current_heading > yaw_angle_right:
      deviation_heading = current_heading - yaw_angle_right
      send_attitude_target(yaw_angle=(current_heading - deviation_heading), thrust=0.5)

    if distance_front == distance_front_minimum:
      vehicle.mode = VehicleMode("LOITER")
      print("\n")
      print("[INFO] Vehicle in hold position...")
      print("\n")

    if distance_front < distance_front_minimum:
      vehicle.mode = VehicleMode("LAND")
      print("\n")
      print("[INFO] Vehicle is landing...")
      print("\n")
      disarm_vehicle()

def window_pass_mission_after():
    # # activate only for indoor mission only
    # vehicle.mode = VehicleMode("LOITER")
    # print("\n")
    # print("[INFO] Vehicle in hold position...")
    # print("\n")
    # current_roll = vehicle.attitude.roll
    # current_heading = vehicle.heading
    # if yaw_angle_option == 1:
    #   target_heading = yaw_angle_right
    # if yaw_angle_option == -1:
    #   target_heading = yaw_angle_left
    # if distance_left < distance_left_minimum and distance_right < distance_right_minimum:
    #   print("\n")
    #   print("[INFO] DANGEROUS DISTANCE TO SIDE OBJECTS... !!!")
    #   print("\n")
    #   if current_heading < target_heading:
    #     deviation_heading = target_heading - current_heading
    #     vehicle.mode = VehicleMode("LOITER")
    #     print("\n")
    #     print("[INFO] Vehicle in hold position...")
    #     print("\n")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(yaw_angle=(current_heading + deviation_heading))
    #     vehicle.mode = VehicleMode("LOITER")
    #     print("\n")
    #     print("[INFO] Vehicle in hold position...")
    #     print("\n")
    #     print("\n")
    #     print("[INFO] Vehicle in hold position...")
    #     print("\n")
    #   elif current_heading > target_heading:      
    #     deviation_heading = current_heading - target_heading
    #     vehicle.mode = VehicleMode("LOITER")
    #     print("\n")
    #     print("[INFO] Vehicle in hold position...")
    #     print("\n")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(yaw_angle=(current_heading - deviation_heading))
    #     vehicle.mode = VehicleMode("LOITER")
    #     print("\n")
    #     print("[INFO] Vehicle in hold position...")
    #     print("\n")
    #   if current_roll < target_roll:      
    #     deviation_roll = target_roll - current_roll
    #     vehicle.mode = VehicleMode("LOITER")
    #     print("\n")
    #     print("[INFO] Vehicle in hold position...")
    #     print("\n")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(roll_angle=(current_roll + deviation_roll))
    #     vehicle.mode = VehicleMode("LOITER")
    #     print("\n")
    #     print("[INFO] Vehicle in hold position...")
    #     print("\n")
    #   elif current_roll > target_roll:
    #     deviation_roll = current_roll - target_roll
    #     vehicle.mode = VehicleMode("LOITER")
    #     print("\n")
    #     print("[INFO] Vehicle in hold position...")
    #     print("\n")
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(roll_angle=(current_roll - deviation_roll))
    #     vehicle.mode = VehicleMode("LOITER")
    #     print("\n")
    #     print("[INFO] Vehicle in hold position...")
    #     print("\n")
    
    # # activate only for indoor mission only
    # if distance_front < distance_front_minimum:
    #     vehicle.mode = VehicleMode("GUIDED")
    #     send_attitude_target(pitch_angle=+5, thrust=0.5)
    #     send_attitude_target(pitch_angle=-5, thrust=0.5)
    #     send_attitude_target(roll_angle=0, thrust=0.5)
    #     vehicle.mode = VehicleMode("LOITER")
    #     if distance_front < distance_front_minimum:
    #         send_attitude_target(pitch_angle=+5, thrust=0.5)
    #         send_attitude_target(roll_angle=-5, thrust=0.5)
    #         send_attitude_target(roll_angle=0, thrust=0.5)
    #         vehicle.mode = VehicleMode("LOITER")
    #         if distance_front > distance_front_minimum:
    #             vehicle.mode = VehicleMode("AUTO")
    #             vehicle.commands.next = window_forward_waypoint_after # going forward to window location
    #             print("\n")
    #             print("[INFO] Current Waypoint: ", vehicle.commands.next)
    #             print("\n")
    #             if vehicle.commands.next == window_forward_waypoint_after:
    #                 print("\n")
    #                 print("[INFO] Vehicle in window_forward_waypoint_after...")
    #                 print("\n")

    #             vehicle.mode = VehicleMode("LOITER")
    #             print("\n")
    #             print("[INFO] Vehicle in hold position...")
    #             print("\n")

    #     if distance_front > distance_front_minimum and distance_left >= distance_left_minimum and distance_right >= distance_right_minimum:
    #         vehicle.mode = VehicleMode("AUTO")
    #         vehicle.commands.next = window_forward_waypoint_after # going forward to window location
    #         print("\n")
    #         print("[INFO] Current Waypoint: ", vehicle.commands.next)
    #         print("\n")

    #         vehicle.mode = VehicleMode("LOITER")
    #         print("\n")
    #         print("[INFO] Vehicle in hold position...")
    #         print("\n")

    #     if distance_front > distance_front_minimum and distance_left < distance_left_minimum and distance_right < distance_right_minimum:
    #         vehicle.mode = VehicleMode("GUIDED")
    #         send_attitude_target(pitch_angle=-5, thrust=0.5)
    #         send_attitude_target(pitch_angle=+5, thrust=0.5)
    #         send_attitude_target(pitch_angle=0, thrust=0.5)
    #         vehicle.mode = VehicleMode("LOITER")
    #         print("\n")
    #         print("[INFO] Vehicle in hold position...")
    #         print("\n")

    #         if distance_front > distance_front_minimum and distance_left < distance_left_minimum and distance_right < distance_right_minimum:
    #             vehicle.mode = VehicleMode("GUIDED")
    #             send_attitude_target(pitch_angle=+5, thrust=0.5)
    #             send_attitude_target(pitch_angle=-5, thrust=0.5)
    #             send_attitude_target(pitch_angle=0, thrust=0.5)
    #             vehicle.mode = VehicleMode("LOITER")
    #             print("\n")
    #             print("[INFO] Vehicle in hold position...")
    #             print("\n")
    #             if distance_front > distance_front_minimum and distance_left >= distance_left_minimum and distance_right >= distance_right_minimum:
    #                 vehicle.mode = VehicleMode("AUTO")
    #                 vehicle.commands.next = window_forward_waypoint_after # going forward to window location
    #                 print("\n")
    #                 print("[INFO] Current Waypoint: ", vehicle.commands.next)
    #                 print("\n")
    #                 if vehicle.commands.next == window_forward_waypoint_after:
    #                     print("\n")
    #                     print("[INFO] Vehicle in window_forward_waypoint_after...")
    #                     print("\n")

    #                 vehicle.mode = VehicleMode("LOITER")
    #                 print("\n")
    #                 print("[INFO] Vehicle in hold position...")
    #                 print("\n")

    #         if distance_front > distance_front_minimum and distance_left >= distance_left_minimum and distance_right >= distance_right_minimum:
    #             vehicle.mode = VehicleMode("AUTO")
    #             vehicle.commands.next = window_forward_waypoint_after # going forward to window location
    #             print("\n")
    #             print("[INFO] Current Waypoint: ", vehicle.commands.next)
    #             print("\n")
    #             if vehicle.commands.next == window_forward_waypoint_after:
    #                 print("\n")
    #                 print("[INFO] Vehicle in window_forward_waypoint_after...")
    #                 print("\n")

    #             vehicle.mode = VehicleMode("LOITER")
    #             print("\n")
    #             print("[INFO] Vehicle in hold position...")
    #             print("\n")

    
    # print("\n")
    # print("[INFO] Going to window position...")
    # print("\n")
 
    # vehicle.commands.next = window_loiter_waypoint_after # going forward to window location
    # print("\n")
    # print("[INFO] Current Waypoint: ", vehicle.commands.next)
    # print("\n")
    # if vehicle.commands.next == window_loiter_waypoint_after:
    #     print("\n")
    #     print("[INFO] Vehicle in window_loiter_waypoint_after...")
    #     print("\n")

    # vehicle.mode = VehicleMode("LOITER")
    # print("\n")
    # print("[INFO] Vehicle in hold position...")
    # print("\n")

    # activate only for indoor and outdoor mission combination
    vehicle.mode = VehicleMode("LOITER")
    print("\n")
    print("[INFO] Vehicle in hold position...")
    print("\n")

    vehicle.mode = VehicleMode("GUIDED")
    current_roll = vehicle.attitude.roll
    current_heading = vehicle.heading
    if yaw_angle_option == -1:
      send_attitude_target(pitch_angle=0, yaw_angle = yaw_angle_left, thrust = 0.5)
      if current_roll < target_roll:
         deviation_roll = target_roll - current_roll
         send_attitude_target(roll_angle=(current_roll + deviation_roll), thrust=0.5)
      elif current_roll > target_roll:
         deviation_roll = current_roll - target_roll
         send_attitude_target(roll_angle=(current_roll - deviation_roll), thrust=0.5)
        
    if yaw_angle_option == 1:
      send_attitude_target(pitch_angle=0, yaw_angle = yaw_angle_right, thrust = 0.5)
      if current_roll < target_roll:
        deviation_roll = target_roll - current_roll
        send_attitude_target(roll_angle=(current_roll + deviation_roll), thrust=0.5)
      elif current_roll > target_roll:
        deviation_roll = current_roll - target_roll
        send_attitude_target(roll_angle=(current_roll - deviation_roll), thrust=0.5)

    # moving forward to the window position
    print("\n")
    print("[INFO] Moving forward to window position")
    print("\n")
    if vehicle.location.global_relative_frame.alt == altitude_drop:
        send_attitude_target(pitch_angle = 0, thrust = 0.5)
        if distance_front < distance_front_minimum:
            vehicle.mode = VehicleMode("GUIDED")
            send_attitude_target(pitch_angle=+5, thrust=0.5)
            send_attitude_target(pitch_angle=-5, thrust=0.5)
            send_attitude_target(roll_angle=-5, thrust=0.5)
            vehicle.mode = VehicleMode("LOITER")
            if distance_front < distance_front_minimum:
                send_attitude_target(pitch_angle=+5, thrust=0.5)
                send_attitude_target(roll_angle=+5, thrust=0.5)
                send_attitude_target(roll_angle=-5, thrust=0.5)
                vehicle.mode = VehicleMode("LOITER")
                if distance_front > distance_front_minimum:
                    send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_window)
                    vehicle.mode = VehicleMode("LOITER")
        if distance_front > distance_front_minimum and distance_left >= distance_left_minimum and distance_right >= distance_right_minimum:
            send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_window)
            vehicle.mode = VehicleMode("LOITER")
        if distance_front > distance_front_minimum and distance_left < distance_left_minimum and distance_right < distance_right_minimum:
            vehicle.mode = VehicleMode("GUIDED")
            send_attitude_target(pitch_angle=-5, thrust=0.5)
            send_attitude_target(pitch_angle=+5, thrust=0.5)
            vehicle.mode = VehicleMode("LOITER")
            if distance_front > distance_front_minimum and distance_left < distance_left_minimum and distance_right < distance_right_minimum:
                vehicle.mode = VehicleMode("GUIDED")
                send_attitude_target(pitch_angle=+5, thrust=0.5)
                send_attitude_target(pitch_angle=-5, thrust=0.5)
                vehicle.mode = VehicleMode("LOITER")
                if distance_front > distance_front_minimum and distance_left >= distance_left_minimum and distance_right >= distance_right_minimum:
                    send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_window)
                    vehicle.mode = VehicleMode("LOITER")
            if distance_front > distance_front_minimum and distance_left >= distance_left_minimum and distance_right >= distance_right_minimum:
                send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_window)
                vehicle.mode = VehicleMode("LOITER")
    
    if vehicle.location.global_relative_frame.alt <= altitude_drop:
        send_attitude_target(pitch_angle = 0, thrust = 0.7)
        if vehicle.location.global_relative_frame.alt == altitude_drop:
            send_attitude_target(pitch_angle = 0, thrust = 0.5)    
            send_attitude_target(pitch_angle = 0, thrust = 0.5)
        if distance_front < distance_front_minimum:
            vehicle.mode = VehicleMode("GUIDED")
            send_attitude_target(pitch_angle=+5, thrust=0.5)
            send_attitude_target(pitch_angle=-5, thrust=0.5)
            send_attitude_target(roll_angle=-5, thrust=0.5)
            vehicle.mode = VehicleMode("LOITER")
            if distance_front < distance_front_minimum:
                send_attitude_target(pitch_angle=+5, thrust=0.5)
                send_attitude_target(roll_angle=+5, thrust=0.5)
                send_attitude_target(roll_angle=-5, thrust=0.5)
                vehicle.mode = VehicleMode("LOITER")
                if distance_front > distance_front_minimum:
                    send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_window)
                    vehicle.mode = VehicleMode("LOITER")
        if distance_front > distance_front_minimum and distance_left >= distance_left_minimum and distance_right >= distance_right_minimum:
            send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_window)
            vehicle.mode = VehicleMode("LOITER")
        if distance_front > distance_front_minimum and distance_left < distance_left_minimum and distance_right < distance_right_minimum:
            vehicle.mode = VehicleMode("GUIDED")
            send_attitude_target(pitch_angle=-5, thrust=0.5)
            send_attitude_target(pitch_angle=+5, thrust=0.5)
            vehicle.mode = VehicleMode("LOITER")
            if distance_front > distance_front_minimum and distance_left < distance_left_minimum and distance_right < distance_right_minimum:
                vehicle.mode = VehicleMode("GUIDED")
                send_attitude_target(pitch_angle=+5, thrust=0.5)
                send_attitude_target(pitch_angle=-5, thrust=0.5)
                vehicle.mode = VehicleMode("LOITER")
                if distance_front > distance_front_minimum and distance_left >= distance_left_minimum and distance_right >= distance_right_minimum:
                    send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_window)
                    vehicle.mode = VehicleMode("LOITER")
            if distance_front > distance_front_minimum and distance_left >= distance_left_minimum and distance_right >= distance_right_minimum:
                send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_window)
                vehicle.mode = VehicleMode("LOITER")
          
    if vehicle.location.global_relative_frame.alt >= altitude_drop:
        send_attitude_target(pitch_angle = 0, thrust = 0.3)
        if vehicle.location.global_relative_frame.alt == altitude_drop:
            send_attitude_target(pitch_angle = 0, thrust = 0.5)
            send_attitude_target(pitch_angle = 0, thrust = 0.5)
        if distance_front < distance_front_minimum:
            vehicle.mode = VehicleMode("GUIDED")
            send_attitude_target(pitch_angle=+5, thrust=0.5)
            send_attitude_target(pitch_angle=-5, thrust=0.5)
            send_attitude_target(roll_angle=-5, thrust=0.5)
            vehicle.mode = VehicleMode("LOITER")
            if distance_front < distance_front_minimum:
                send_attitude_target(pitch_angle=+5, thrust=0.5)
                send_attitude_target(roll_angle=+5, thrust=0.5)
                send_attitude_target(roll_angle=-5, thrust=0.5)
                vehicle.mode = VehicleMode("LOITER")
                if distance_front > distance_front_minimum:
                    send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_window)
                    vehicle.mode = VehicleMode("LOITER")
        if distance_front > distance_front_minimum and distance_left >= distance_left_minimum and distance_right >= distance_right_minimum:
            send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_window)
            vehicle.mode = VehicleMode("LOITER")
        if distance_front > distance_front_minimum and distance_left < distance_left_minimum and distance_right < distance_right_minimum:
            vehicle.mode = VehicleMode("GUIDED")
            send_attitude_target(pitch_angle=-5, thrust=0.5)
            send_attitude_target(pitch_angle=+5, thrust=0.5)
            vehicle.mode = VehicleMode("LOITER")
            if distance_front > distance_front_minimum and distance_left < distance_left_minimum and distance_right < distance_right_minimum:
                vehicle.mode = VehicleMode("GUIDED")
                send_attitude_target(pitch_angle=+5, thrust=0.5)
                send_attitude_target(pitch_angle=-5, thrust=0.5)
                vehicle.mode = VehicleMode("LOITER")
                if distance_front > distance_front_minimum and distance_left >= distance_left_minimum and distance_right >= distance_right_minimum:
                    send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_window)
                    vehicle.mode = VehicleMode("LOITER")
            if distance_front > distance_front_minimum and distance_left >= distance_left_minimum and distance_right >= distance_right_minimum:
                send_attitude_target(pitch_angle = -5, thrust = 0.5, duration = forward_duration_window)
                vehicle.mode = VehicleMode("LOITER")
          
    if current_roll < target_roll:
        deviation_roll = target_roll - current_roll
        send_attitude_target(roll_angle=(current_roll + deviation_roll), thrust=0.5)
    elif current_roll > target_roll:
        deviation_roll = current_roll - target_roll
        send_attitude_target(roll_angle=(current_roll - deviation_roll), thrust=0.5)
    if current_heading < yaw_angle_right:
        deviation_heading = yaw_angle_right - current_heading
        send_attitude_target(yaw_angle=(current_heading + deviation_heading), thrust=0.5)
    elif current_heading > yaw_angle_right:
        deviation_heading = current_heading - yaw_angle_right
        send_attitude_target(yaw_angle=(current_heading - deviation_heading), thrust=0.5)

    # if distance_back == distance_back_minimum:
    #     vehicle.mode = VehicleMode("LOITER")
    #     print("\n")
    #     print("[INFO] Vehicle in hold position...")
    #     print("\n")

    # if distance_back < distance_back_minimum:
    #     vehicle.mode = VehicleMode("LAND")
    #     disarm_vehicle()

def outdoor_payload_first():
    # activate for outdoor mission only
    if distance_back == distance_back_minimum:
        vehicle.mode = VehicleMode("LOITER")
        print("\n")
        print("[INFO] Vehicle in hold position...")
        print("\n")

    if distance_back < distance_back_minimum:
        vehicle.mode = VehicleMode("LAND")
        disarm_vehicle()

    vehicle.mode = VehicleMode("AUTO")

    print("\n")
    print("[INFO] Going to outdoor first payload drop position...")
    print("\n")
    
    if yaw_angle_option == -1:
        vehicle.commands.next = outdoor_drop_mission_position_1_forward_waypoint_left # going forward to outdoor first payload drop position left side
        print("\n")
        print("[INFO] Current Waypoint: ", vehicle.commands.next)
        print("\n")
        if vehicle.commands.next == outdoor_drop_mission_position_1_forward_waypoint_left:
            print("\n")
            print("[INFO] Vehicle in outdoor_drop_mission_position_1_forward_waypoint_left...")
            print("\n")

        vehicle.commands.next = outdoor_drop_mission_position_1_loiter_waypoint_left # arrived in outdoor first payload drop position left side
        print("\n")
        print("[INFO] Current Waypoint: ", vehicle.commands.next)
        print("\n")
        if vehicle.commands.next == outdoor_drop_mission_position_1_loiter_waypoint_left:
            print("\n")
            print("[INFO] Vehicle in outdoor_drop_mission_position_1_loiter_waypoint_left...")
            print("\n")

        vehicle.mode = VehicleMode("LOITER")
        print("\n")
        print("[INFO] Vehicle in hold position...")
        print("\n")
    
    if yaw_angle_option == 1:
        vehicle.commands.next = outdoor_drop_mission_position_1_forward_waypoint_right # going forward to outdoor first payload drop position right side
        print("\n")
        print("[INFO] Current Waypoint: ", vehicle.commands.next)
        print("\n")
        if vehicle.commands.next == outdoor_drop_mission_position_1_forward_waypoint_right:
            print("\n")
            print("[INFO] Vehicle in outdoor_drop_mission_position_1_forward_waypoint_right...")
            print("\n")

        vehicle.commands.next = outdoor_drop_mission_position_1_loiter_waypoint_right # arrived in outdoor first payload drop position right side
        print("\n")
        print("[INFO] Current Waypoint: ", vehicle.commands.next)
        print("\n")
        if vehicle.commands.next == outdoor_drop_mission_position_1_forward_waypoint_left:
            print("\n")
            print("[INFO] Vehicle in outdoor_drop_mission_position_1_loiter_waypoint_right...")
            print("\n")

        vehicle.mode = VehicleMode("LOITER")
        print("\n")
        print("[INFO] Vehicle in hold position...")
        print("\n")

def outdoor_payload_second():
    # activate for outdoor mission only
    vehicle.mode = VehicleMode("LOITER")
    print("\n")
    print("[INFO] Vehicle in hold position...")
    print("\n")

    vehicle.mode = VehicleMode("AUTO")

    print("\n")
    print("[INFO] Going to outdoor second payload drop position...")
    print("\n")

    if yaw_angle_option == -1:
        vehicle.commands.next = outdoor_drop_mission_position_2_forward_waypoint_left # going forward to outdoor second payload drop position left side
        print("\n")
        print("[INFO] Current Waypoint: ", vehicle.commands.next)
        print("\n")
        if vehicle.commands.next == outdoor_drop_mission_position_2_forward_waypoint_left:
            print("\n")
            print("[INFO] Vehicle in outdoor_drop_mission_position_2_forward_waypoint_left...")
            print("\n")

        vehicle.commands.next = outdoor_drop_mission_position_2_loiter_waypoint_left # arrived in outdoor second payload drop position left side
        print("\n")
        print("[INFO] Current Waypoint: ", vehicle.commands.next)
        print("\n")
        if vehicle.commands.next == outdoor_drop_mission_position_2_forward_waypoint_left:
            print("\n")
            print("[INFO] Vehicle in outdoor_drop_mission_position_2_forward_waypoint_left...")
            print("\n")

        vehicle.mode = VehicleMode("LOITER")
        print("\n")
        print("[INFO] Vehicle in hold position...")
        print("\n")

    if yaw_angle_option == 1:
        vehicle.commands.next = outdoor_drop_mission_position_2_forward_waypoint_right # going forward to outdoor second payload drop position right side
        print("\n")
        print("[INFO] Current Waypoint: ", vehicle.commands.next)
        print("\n")
        if vehicle.commands.next == outdoor_drop_mission_position_2_forward_waypoint_right:
            print("\n")
            print("[INFO] Vehicle in outdoor_drop_mission_position_2_forward_waypoint_right...")
            print("\n")

        vehicle.commands.next = outdoor_drop_mission_position_2_loiter_waypoint_right # arrived in outdoor second payload drop position right side
        print("\n")
        print("[INFO] Current Waypoint: ", vehicle.commands.next)
        print("\n")
        if vehicle.commands.next == outdoor_drop_mission_position_2_forward_waypoint_right:
            print("\n")
            print("[INFO] Vehicle in outdoor_drop_mission_position_2_forward_waypoint_right...")
            print("\n")

        vehicle.mode = VehicleMode("LOITER")
        print("\n")
        print("[INFO] Vehicle in hold position...")
        print("\n")

def going_to_new_land_position_mission():
    # activate only on outdoor mission
    vehicle.mode = VehicleMode("LOITER")
    print("\n")
    print("[INFO] Vehicle in hold position...")
    print("\n")

    vehicle.mode = VehicleMode("AUTO")

    print("\n")
    print("[INFO] Going to new landing zone position...")
    print("\n")

    if yaw_angle_option == -1:
        vehicle.commands.next = going_new_land_forward_waypoint_left # going forward to new landing zone position left side
        print("\n")
        print("[INFO] Current Waypoint: ", vehicle.commands.next)
        print("\n")
        if vehicle.commands.next == going_new_land_forward_waypoint_left:
            print("\n")
            print("[INFO] Vehicle in going_new_land_forward_waypoint_left...")
            print("\n")

        vehicle.commands.next = going_new_land_loiter_waypoint_left # going forward to new landing zone position left side
        print("\n")
        print("[INFO] Current Waypoint: ", vehicle.commands.next)
        print("\n")
        if vehicle.commands.next == going_new_land_loiter_waypoint_left:
            print("\n")
            print("[INFO] Vehicle in going_new_land_loiter_waypoint_left...")
            print("\n")

        vehicle.mode = VehicleMode("LOITER")
        print("\n")
        print("[INFO] Vehicle in hold position...")
        print("\n")

    if yaw_angle_option == 1:
        vehicle.commands.next = going_new_land_forward_waypoint_right # going forward to new landing zone position right side
        print("\n")
        print("[INFO] Current Waypoint: ", vehicle.commands.next)
        print("\n")
        if vehicle.commands.next == going_new_land_forward_waypoint_right:
            print("\n")
            print("[INFO] Vehicle in going_new_land_forward_waypoint_right...")
            print("\n")

        vehicle.commands.next = going_new_land_loiter_waypoint_right # going forward to new landing zone position right side
        print("\n")
        print("[INFO] Current Waypoint: ", vehicle.commands.next)
        print("\n")
        if vehicle.commands.next == going_new_land_loiter_waypoint_right:
            print("\n")
            print("[INFO] Vehicle in going_new_land_loiter_waypoint_right...")
            print("\n")

        vehicle.mode = VehicleMode("LOITER")
        print("\n")
        print("[INFO] Vehicle in hold position...")
        print("\n")

def servo_gripper_start():
    # activate for indoor mission
    servo_gripper_left.angle = gripper_left_servo_angle_start
    servo_gripper_right.angle = gripper_right_servo_angle_start
    time.sleep(1)
    print("\n")
    print("[INFO] Gripper servos are ready...")
    print("\n")

def servo_gripper_pickup():
    # activate for indoor mission
    time.sleep(5) # wait for 5 seconds
    servo_gripper_left.angle = gripper_left_servo_angle_final
    servo_gripper_right.angle = gripper_right_servo_angle_final
    time.sleep(1)
    print("\n")
    print("[INFO] Gripper servos are picking up the payload...")
    print("\n")

def servo_gripper_release():
    # activate for indoor mission
    servo_gripper_left.angle = gripper_left_servo_angle_release
    servo_gripper_right.angle = gripper_right_servo_angle_release
    time.sleep(1)
    print("\n")
    print("[INFO] Gripper servos are release the payload...")
    print("\n")

def front_cam_servo_forward():
    # activate for indoor and or outdoor missions
    servo_gripper_front_cam.angle = front_cam_servo_angle_front
    time.sleep(1)
    print("\n")
    print("[INFO] Front camera servos is facing forward...")
    print("\n")

def front_cam_servo_bottom():
    # activate for indoor and or outdoor missions
    servo_gripper_front_cam.angle = front_cam_servo_angle_bottom
    time.sleep(1)
    print("\n")
    print("[INFO] Front camera servos is facing bottom...")
    print("\n")

def servo_payload_left_right_start():
    # activate for outdoor mission
    servo_payload_left.angle = payload_drop_servo_left_start
    servo_payload_right.angle = payload_drop_servo_right_start
    time.sleep(1)
    print("\n")
    print("[INFO] Payload drop left and right servos are ready...")
    print("\n")

def servo_payload_left_release():
    # activate for outdoor mission
    servo_payload_left.angle = payload_drop_servo_left_final
    time.sleep(1)
    print("\n")
    print("[INFO] Outdoor first payload is release...")
    print("\n")

def servo_payload_right_release():
    # activate for outdoor mission
    servo_payload_left.angle = payload_drop_servo_right_final
    time.sleep(1)
    print("\n")
    print("[INFO] Outdoor second payload is release...")
    print("\n")

def bucket_color_detection_load():
    # qctivate for inddor mission also on indoor and outdoor mission combination
    print("\n")
    print("[INFO] LOADING THE BUCKET COLOR DETECTION kNN-BASED SYSTEM PARAMETERS...")
    print("\n")

    # color detection algorithm
    # checking whether the training data is ready
    PATH = './training.data'

    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        print("\n")
        print ("[INFO] Color detection training data is ready, color detection classifier is loading...")
        print("\n")
    else:
        print("\n")
        print ("[INFO] Color detection training data is being created...")
        print("\n")

    open('training.data', 'w')
    color_histogram_feature_extraction.training()
    print("\n")
    print ("[INFO] Color detecton training data is ready, color detection classifier is loading...")
    print("\n")

def find_marker():
    # convert the image to grayscale, blur it, and detect edges
    cam_front = cap_front()
    _, frame_front = cam_front.read()
    gray_front = cv2.cvtColor(frame_front, cv2.COLOR_BGR2GRAY)
    gray_front = cv2.GaussianBlur(gray_front, (5, 5), 0)
    edged_front = cv2.Canny(gray_front, 35, 125)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
    cnts_front = cv2.findContours(edged_front.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts_front = imutils.grab_contours(cnts_front)
    c_front = max(cnts_front, key = cv2.contourArea)
	# compute the bounding box of the of the paper region and return it
    return cv2.minAreaRect(c_front)

# find distance from camera to object
def distance_to_camera(known_width,  focal_length, per_width):
	# compute and return the distance from the maker to the camera
	return (known_width * focal_length) / per_width

# def new_home_location():
#     # Set vehicle home_location, mode, and armed attributes (the only settable attributes)

#     print("\n[INFO] Set new home location")
#     # Home location must be within 50km of EKF home location (or setting will fail silently)
#     # In this case, just set value to current location with an easily recognisable altitude (0)
#     my_location_alt = vehicle.location.global_frame
#     my_location_alt.alt = 0
#     vehicle.home_location = my_location_alt
#     print("[INFO] New Home Location (from attribute - altitude should be 0): %s" % vehicle.home_location)

#     # Confirm current value on vehicle by re-downloading commands
#     cmds = vehicle.commands
#     cmds.download()
#     cmds.wait_ready()
#     print("[INFO] Checking New Home Location (from vehicle - altitude should be 0): %s" % vehicle.home_location)

# def servo_landing():
#     msg_servo_landing_left = vehcile.message.factory.comment_long_encode(
#         0, 0, # target_system, target_comnponent
#         mavuitl.mavlink.MAV_CMD_DO_SET_SERVO # command
#         0, # confirmation
#         7, # servo_number
#         1000, # servo position between 1000 and 2000
#         0, 0, 0, 0, 0) # param 3~ 7 not used
        
#      msg_servo_landing_right = vehcile.message.factory.comment_long_encode(
#         0, 0, # target_system, target_comnponent
#         mavuitl.mavlink.MAV_CMD_DO_SET_SERVO # command
#         0, # confirmation
#         8, # servo_number
#         1000, # servo position between 1000 and 2000
#         0, 0, 0, 0, 0) # param 3~ 7 not used
    
#     # send command to vehicle
#     vehicle.send_mavlink(msg_servo_landing_left)
#     vehicle.send_mavlink(msg_servo_landing_right)

# def servo_takeoff():
#     msg_servo_takeoff_left = vehcile.message.factory.comment_long_encode(
#         0, 0, # target_system, target_comnponent
#         mavuitl.mavlink.MAV_CMD_DO_SET_SERVO # command
#         0, # confirmation
#         7, # servo_number
#         2000, # servo position between 1000 and 2000
#         0, 0, 0, 0, 0) # param 3~ 7 not used
        
#      msg_servo_takeoff_right = vehcile.message.factory.comment_long_encode(
#         0, 0, # target_system, target_comnponent
#         mavuitl.mavlink.MAV_CMD_DO_SET_SERVO # command
#         0, # confirmation
#         8, # servo_number
#         2000, # servo position between 1000 and 2000
#         0, 0, 0, 0, 0) # param 3~ 7 not used
    
#      # send command to vehicle
#     vehicle.send_mavlink(msg_servo_takeoff_left)
#     vehicle.send_mavlink(msg_servo_takeoff_right)
   
# ====================================================================================================================================================================================== #

# Object detection system #

# ====================================================================================================================================================================================== #

print("\n")
print("[INFO] OBJECT DETECTION SYSTEM IS STARTING...")
print("\n")

print("\n")
print("[INFO] LOADING THE OBJECT DETECTION SYSTEM PARAMETERS...")
print("\n")

# Load the YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False) 
model.conf = threshold
model.max_det = max_detect_object
model.multi_label = True

# capture the images from the cameras, allow the camera sensor to warmup, and start the FPS counter
cap_bottom = cv2.VideoCapture(bottom_camera)
cap_bottom.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap_bottom.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
cap_bottom.set(cv2.CAP_PROP_FPS, 2)

cap_front = cv2.VideoCapture(front_camera)
cap_front.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap_front.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
cap_front.set(cv2.CAP_PROP_FPS, 2)

if cap_front == 'NoneType':
  # raise ValueError("ERROR! Unable to open video source(s).")
    print("\n")
    print("[INFO] Front camera not detected")
    print("\n")
    land_mission()
    disarm_vehicle()
    print("[INFO] System is Rebooting...")
    print("\n")
    os.system("sudo reboot") # reboot the Ubuntu OS on Raspberry Pi

if cap_bottom == 'NoneType':
  # raise ValueError("ERROR! Unable to open video source(s).")
    print("\n")
    print("[INFO] Bottom camera not detected")
    print("\n")
    land_mission()
    disarm_vehicle()
    print("[INFO] System is Rebooting...")
    print("\n")
    os.system("sudo reboot") # reboot the Ubuntu OS on Raspberry Pi

# code foe webcam streaming
# front_camera
stream_front = Stream("front_camera", size=(480, 480), quality=50, fps=10)
server_front = MjpegServer(localhost, port_front)
server_front.add_stream(stream_front)
server_front.start()

# bottom_camera
stream_bottom = Stream("bottom_camera", size=(480, 480), quality=50, fps=10)
server_bottom = MjpegServer(localhost, port_bottom)
server_bottom.add_stream(stream_bottom)
server_bottom.start()

fps = FPS().start()

print("\n")
print("[INFO] CAMERA IS STARTING...")
print("\n")

print("\n")
print("[INFO] SYSTEM IS READY...")
print("\n")

# ====================================================================================================================================================================================== #

# Main program #

# ====================================================================================================================================================================================== #

def main_code():
       
    # time.sleep(wait_time)

    print("====================================================================================================================================================")
    print("\n")
    print("[INFO] INDOOR MISSION IS STARTING...")
    print("\n")
    print("====================================================================================================================================================")
    
    target_heading = start_heading
    current_roll = vehicle.attitude.roll
    print("[INFO] Target heading : ", target_heading)
    print("\n")
    battery_level()
    vehicle_altitude()
    # Take off 1m in GUIDED mode.
    # ch_6_input = vehicle.channels['6']
    current_heading = vehicle.heading
    print("[INFO] Current heading : ", current_heading)
    print("\n")

    front_cam_servo_bottom()

    servo_gripper_start()
    
    if not vehicle.home_location :
        print("\n")
        print("[INFO] Vehicle not in home position")
        print("\n")
    # print("[INFO] Waiting for the Channel 6 / GEAR switch to start the mission")
    # print("\n")
    # if ch_6_input > 1000 and vehicle.home_location and current_heading == start_heading :
    if vehicle.home_location and current_heading == start_heading :
        arm_and_takeoff(altitude_indoor)
        vehicle_altitude()
    if vehicle.location.global_relative_frame.alt == altitude_indoor:
        vehicle_altitude()
        vehicle.mode = VehicleMode("LOITER")
        print("[INFO] Vehicle is hold position...")
        print("\n")
        vehicle_altitude()
    if vehicle.location.global_relative_frame.alt < altitude_indoor:
        vehicle_altitude()
        vehicle.mode = VehicleMode("GUIDED")
        send_attitude_target(pitch_angle= 0, yaw_angle= start_heading, thrust=0.7)
        vehicle_altitude()
        if vehicle.location.global_relative_frame.alt == altitude_indoor:
          vehicle_altitude()
          vehicle.mode = VehicleMode("LOITER")
          print("[INFO] Vehicle is hold position...")
          print("\n")
          vehicle_altitude()
    if vehicle.location.global_relative_frame.alt > altitude_indoor:
        vehicle_altitude()
        vehicle.mode = VehicleMode("GUIDED")
        send_attitude_target(pitch_angle= 0, yaw_angle= start_heading, thrust=0.3)
        vehicle_altitude()
        if vehicle.location.global_relative_frame.alt == altitude_indoor:
          vehicle_altitude()
          vehicle.mode = VehicleMode("LOITER")
          print("[INFO] Vehicle is hold position...")
          print("\n")
          vehicle_altitude()
        
    battery_level()    

    # going forward
    forward_payload_mission()

    # turn on the magnetic gripper
    magnetic_gripper.ChangeDutyCycle(75) # activate the magnetic gripper pickup command
    print("\n")
    print("[INFO] Magnetic gripper is picking up the payload...")
    print("\n")

    # loop over some frames...this time using the threaded stream
    while fps._numFrames < args["num_frames"]:

        # Bottom Camera
        _, frame_bottom = cap_bottom.read()
        # frame_bottom = imutils.resize(frame_bottom, height=320, width=320)

        print("\n")

        # start the webcam streaming for bottom camera
        stream_bottom.set_frame(frame_bottom)

        print("\n")

        # Front Camera
        _, frame_front = cap_front.read()
        # frame_front = imutils.resize(frame_bottom, height=320, width=320)

        print("\n")

        # start the webcam streaming for front camera
        stream_front.set_frame(frame_front)

        print("\n")

        # calculating distance from camera to object in loop
        marker = find_marker(frame_front)
        distances_milimeters = distance_to_camera(known_width, focal_length, marker[1][0])
        print("\n")
        print("[INFO] Distances from Vehicle to Front Object in Meter(s): ", round(distances_milimeters*100, 2))
        print("\n")

        # Run YOLOv5 inference on the bottom frame
        results_bottom = model(frame_bottom)
        
        result_df_bottom = pd.DataFrame(results_bottom.pandas().xyxy[0])

        object_bottom = ""
        score_bottom = 0
        xmin_bottom = 0
        xmax_bottom = 0
        ymin_bottom = 0
        ymax_bottom = 0

        result_xmin_bottom = (result_df_bottom['xmin'].values)
        result_xmax_bottom = (result_df_bottom['xmax'].values)
        result_ymin_bottom = (result_df_bottom['ymin'].values)
        result_ymax_bottom = (result_df_bottom['ymax'].values)
        
        for xmin_bottom in result_xmin_bottom:
            int(xmin_bottom)

        for xmax_bottom in result_xmax_bottom:
            int(xmax_bottom)

        for ymin_bottom in result_ymin_bottom:
            int(ymin_bottom)
        
        for ymax_bottom in result_ymax_bottom:
            int(ymax_bottom)

        for object_bottom in result_df_bottom['name'].values:
            str(object_bottom)

        for score_bottom in result_df_bottom['confidence'].values:
            float(score_bottom)
        
        # show the results        
        if result_df_bottom.empty == True :
            print("\n")
            print('[INFO] Bottom camera - no object detected')
        else :
            # print('Object: ', result_df['name'].to_frame().to_csv(index=False, header=False)rint('Score: ', result_df['confidence'].to_frame().to_csv(index=False, header=False))
            print("\n")
            print('[INFO] Bottom camera - captured object: ', object_bottom, '|', 'Score: {:.2f}'.format(score_bottom*100), '%')
            print("\n")

        # Front Camera
        # frame_front = imutils.resize(frame_front, height=320, width=320)
        marker_front = find_marker(frame_front)
        focal_length = ((marker_front[1][0] * known_distance) / known_width)

        # Run YOLOv5 inference on the front frame
        results_front = model(frame_front)
        
        result_df_front = pd.DataFrame(results_front.pandas().xyxy[0])

        object_front = ""
        score_front = 0
        xmin_front = 0
        xmax_front = 0
        ymin_front = 0
        ymax_front = 0

        result_xmin_front = (result_df_front['xmin'].values)
        result_xmax_front = (result_df_front['xmax'].values)
        result_ymin_front = (result_df_front['ymin'].values)
        result_ymax_front = (result_df_front['ymax'].values)
        
        for xmin_front in result_xmin_front:
            int(xmin_front)

        for xmax_front in result_xmax_front:
            int(xmax_front)

        for ymin_front in result_ymin_front:
            int(ymin_front)
        
        for ymax_front in result_ymax_front:
            int(ymax_front)

        for object_front in result_df_front['name'].values:
            str(object_front)

        for score_front in result_df_front['confidence'].values:
            float(score_front)
        
        # show the results        
        if result_df_front.empty == True :
            print("\n")
            print('[INFO] Front camera - no object detected')
        else :
            # print('Object: ', result_df['name'].to_frame().to_csv(index=False, header=False)rint('Score: ', result_df['confidence'].to_frame().to_csv(index=False, header=False))
            print("\n")
            print('[INFO] Front camera - captured object: ', object_front, '|', 'Score: {:.2f}'.format(score_front*100), '%')
            print("\n")
 
        # update the FPS counter
        fps.update()
       
        # create causal reason for the object detection results
        if result_df_bottom.empty == False or result_df_front.empty == False and object_bottom == object_detected_1 or object_front == object_detected_1 or vehicle.commands.next == payload_loiter_waypoint:
            print("\n")
            print("[INFO] Detection result: ", object_detected_1, "detected")
            print("\n")

            row_size_obj = 40  # pixels
            left_margin_obj = 10  # pixels
            font_size = 1
            font_thickness = 2
            obj_text = 'PAYLOAD INDOOR DETECTED - GRIPPER ON'
            text_location_obj = (left_margin_obj, row_size_obj)
            cv2.putText(frame_bottom, obj_text, text_location_obj, cv2.FONT_HERSHEY_PLAIN, font_size, (255,0,0), font_thickness)

            frame_bottom[:, :, 0] = 0
            frame_bottom[:, :, 0] = 1

            # turn on the magnetic gripper
            magnetic_gripper.ChangeDutyCycle(75) # activate the magnetic gripper pickup command
            print("\n")
            print("[INFO] Magnetic gripper is picking up the payload...")
            print("\n")

            if line_sensor == 0:
                servo_gripper_pickup()

            vehicle_altitude()
            
            battery_level()

            front_cam_servo_forward()

            # moving vehicle forwward to the corner
            forward_corner_misssion()           

            # moving vehicle left or right
            moving_left_right_mission()

            front_cam_servo_bottom()

            if result_df_bottom.empty == False or result_df_front.empty == False and object_bottom == object_detected_2 or object_front == object_detected_2 or vehicle.commands.next == drop_loiter_waypoint:
              print("\n")
              print("[INFO] Detection result: ", object_detected_2 , "detected")
              print("\n")

              row_size_obj = 60  # pixels
              left_margin_obj = 10  # pixels
              font_size = 1
              font_thickness = 2

              obj_text = 'BUCKET BOX DETECTED - GRIPPER OFF'
              text_location_obj = (left_margin_obj, row_size_obj)
              cv2.putText(frame_bottom, obj_text, text_location_obj, cv2.FONT_HERSHEY_PLAIN, font_size, (255,0,255), font_thickness)
            
              frame_bottom[:, :, 0] = 0
              frame_bottom[:, :, 1] = 0
            
              # turn off the magnetic gripper
              magnetic_gripper.ChangeDutyCycle(50) # activate the magnetic gripper release command
              print("\n")
              print("[INFO] Magnetic gripper is releasing the payload...")
              print("\n")

              front_cam_servo_forward()

              window_pass_mission_before()

              if result_df_front == False and object_front == object_detected_3 or vehicle.commands.next == window_forward_waypoint_before_outdoor:
                print("\n")
                print("[INFO] Detection result: ", object_detected_3, "detected")
                print("\n")

                row_size_obj = 40  # pixels
                left_margin_obj = 10  # pixels
                font_size = 1
                font_thickness = 2
                obj_text = 'WINDOW DETECTED - PASSING THROUGH'
                text_location_obj = (left_margin_obj, row_size_obj)
                cv2.putText(frame_front, obj_text, text_location_obj, cv2.FONT_HERSHEY_PLAIN, font_size, (255,0,0), font_thickness)

                frame_front[:, :, 0] = 0
                frame_front[:, :, 0] = 1
              
                window_pass_mission_after()

                # deactivate two lines of codes below for indoor and outdoor mission combination
                # activate only for indoor mission

                land_mission()

                print("\n")
                print("====================================================================================================================================================")
                print("\n")
                print("[INFO] INDOOR MISSION IS FINISHED...")
                print("\n")
                print("====================================================================================================================================================")
                print("\n")
                
                disarm_vehicle()

                print("\n")
                print("====================================================================================================================================================")
                print("\n")
                print("[INFO] VEHICLE IS DISARMED...")
                print("\n")
                print("====================================================================================================================================================")
                print("\n")

                # end of indoor mission

                # start of outdoor mission

                # activate the codes below for indoor and outdoor mission combination

                # print("\n")
                # print("====================================================================================================================================================")
                # print("\n")
                # print("[INFO] OUTDOOR MISSION IS STARTING...")
                # print("\n")
                # print("====================================================================================================================================================")
                # print("\n")

                # front_cam_servo_bottom()
                               
                # going to first outdoor payload mission
                # outdoor_payload_first()

                # # activate the color detection computer vision module to landing the vehicle
                # color_prediction_payload = "n.a."
                # # put text of prediction result on the image frame
                # cv2.putText(frame_front, "Prediction: " + color_prediction, (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, 200, 2)
                # # color histogram feature extraction
                # color_histogram_feature_extraction.color_histogram_of_test_image(frame_front)
                # # showing the object color detection prediction results
                # color_prediction_payload = knn_classifier.main('training.data', 'test.data')

                # if yaw_angle_option == -1:
                #   if color_prediction_payload == payload_outdoor_zone_detected:
                #     vehicle.commands.next = outdoor_drop_mission_position_1_forward_waypoint_left
                #     vehicle.mode = VehicleMode("LOITER")
                #     time.sleep(3)
                #     servo_payload_left_release()
                    
                # if yaw_angle_option == 1:
                #   if color_prediction_payload == payload_outdoor_zone_detected:
                #     vehicle.commands.next = outdoor_drop_mission_position_1_forward_waypoint_right
                #     vehicle.mode = VehicleMode("LOITER")
                #     time.sleep(3)
                #     servo_payload_left_release()
                
                # # going to second outdoor payload mission
                # outdoor_payload_second()

                # if yaw_angle_option == -1:
                #   if color_prediction_payload == payload_outdoor_zone_detected:
                #     vehicle.commands.next = outdoor_drop_mission_position_2_forward_waypoint_left
                #     vehicle.mode = VehicleMode("LOITER")
                #     time.sleep(3)
                #     servo_payload_right_release()
                
                # if yaw_angle_option == 1:
                #   if color_prediction_payload == payload_outdoor_zone_detected:
                #     vehicle.commands.next = outdoor_drop_mission_position_2_forward_waypoint_right
                #     vehicle.mode = VehicleMode("LOITER")
                #     time.sleep(3)
                #     servo_payload_right_release()

                # # going to new landing zone mission
                # going_to_new_land_position_mission()

                # # activate the color detection computer vision module to landing the vehicle
                # color_prediction_landing = "n.a."
                # # put text of prediction result on the image frame
                # cv2.putText(frame_front, "Prediction: " + color_prediction, (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, 200, 2)
                # # color histogram feature extraction
                # color_histogram_feature_extraction.color_histogram_of_test_image(frame_front)
                # # showing the object color detection prediction results
                # color_prediction_landing = knn_classifier.main('training.data', 'test.data')

                # if yaw_angle_option == -1:
                #   if color_prediction_landing == left_landing_zone_detected:
                #     vehicle.commands.next = new_land_loiter_waypoint_left
                #     vehicle.mode = VehicleMode("LOITER")
                #     time.sleep(3)
                #     land_mission()
                #     disarm_vehicle()
                    
                # if yaw_angle_option == 1:
                #   if color_prediction_landing == right_landing_zone_detected:
                #     vehicle.commands.next = new_land_loiter_waypoint_right
                #     vehicle.mode = VehicleMode("LOITER")
                #     time.sleep(3)
                #     land_mission()
                #     disarm_vehicle()

                # # landing and disarm the vehicle
                # land_mission()
                
                # disarm_vehicle()

                # print("\n")
                # print("====================================================================================================================================================")
                # print("\n")
                # print("[INFO] OUTDOOR MISSION IS FINISHED...")
                # print("\n")
                # print("====================================================================================================================================================")
                # print("\n")

                # end of outdoor mission


        # Display the camera frame images
        # cv2.namedWindow("Bottom Camera View", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Bottom Camera View", 640, 480)
        # cv2.imshow("Bottom Camera View", frame_bottom)
        
        # cv2.namedWindow("Front Camera View", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Front Camera View", 640, 480)
        # cv2.imshow("Front Camera View", frame_front)

        fps.stop()

        # calculate FPS
        # seconds = fps.elapsed() - fps
        # fps1 = 1.0 / seconds
        print("\n")
        print("[INFO] Elapsed time in seconds: {:.2f}".format(fps.elapsed()))
        print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
        print("\n")
        
        # put fps info on the frame
        text_color = (255, 255, 0)  # green
        row_size = 20  # pixels
        left_margin = 10  # pixels
        font_size = 1
        font_thickness = 1

        fps_text = 'FPS = {:.2f}'.format(fps.fps())
        text_location = (left_margin, row_size)
        cv2.putText(frame_front, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)
        cv2.putText(frame_front, 'FRONT CAMERA VIEW', (144, 20), cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)

        cv2.putText(frame_front, 'detected : {}'.format(object_front), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, text_color, 1)
        cv2.putText(frame_front, 'score : {:.2f} %'.format(score_front*100), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, text_color, 1)

        cv2.putText(frame_bottom, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)
        cv2.putText(frame_bottom, 'BOTTOM CAMERA VIEW', (144, 20), cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)

        cv2.putText(frame_bottom, 'detected : {}'.format(object_bottom), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, text_color, 1)
        cv2.putText(frame_bottom, 'score : {:.2f} %'.format(score_bottom*100), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, text_color, 1)
        
        # if cv2.waitKey(1) == 13: #13 is the Enter Key
        #     break
        #     print("\n")
        #     print("[INFO] Camera is closed")
        #     print("\n")
        #     print("====================================================================================================================================================")
        
        # program to sending the object detection results as images
        # sender = imagezmq.ImageSender(connect_to=tcp_address)
        # rpi_name = socket.gethostname()
        # try:                
        #     while True:
        #         reply_from_hub = sender.send_image(rpi_name, frame_front)
        #         # main_code()
        # except (KeyboardInterrupt, SystemExit):
        #     pass
        # except Exception as ex:
        #     print('[INFO] Python error with no Exception handler:')
        #     print('[INFO] Traceback error:', ex)
        #     traceback.print_exc()
        # finally:
        #     sender.close()  # close the ZMQ socket and context
        #     sys.exit()
        
        # saving object detection results as video
        # frame_width = 320
        # frame_height = 320
        # frame_size = (frame_width, frame_height)
        # video_save = cv2.VideoWriter('video_stream.avi', cv2.VideoWriter_fourcc(*'MJPEG'), 10, frame_size)
        # video_save.write(frame_front)
        
        print("\n")
        print("====================================================================================================================================================")
        print("\n")
        print("[INFO] ALL MISSION(S) IS COMPLETED...")
        print("\n")
        print("====================================================================================================================================================")

        print("\n")
        print("====================================================================================================================================================")
        print("\n")
        print("[INFO] VEHICLE IS DISARMED, PLEASE TAKE THE VEHICLE...")
        print("\n")
        print("====================================================================================================================================================")
        print("\n")
    
    return result_df_front, result_df_bottom

# wait_key() # wait for the key press to continue the misssion
wait_key()
# main_code()
# Release the video capture object and close the display window
server_front.stop()
server_bottom.stop()
cap_front.release()
cap_bottom.release()
cv2.destroyAllWindows()
