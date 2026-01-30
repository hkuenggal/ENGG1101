#!/usr/bin/env python3
"""
SENSOR DASHBOARD - WITH USB CAMERA FEED & YOLO DETECTION
DHT22 version
"""

import time
import sys
import smbus2
import RPi.GPIO as GPIO
from datetime import datetime
import threading
import json
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import socket
import cv2
import numpy as np

# =================== SENSOR CONFIGURATION ===================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Sensor GPIO pins - DHT22 on GPIO21 (Pin 40)
DHT_PIN = 21          # DHT22 on GPIO21 (Pin 40) - SAME PIN AS DHT11
HC_TRIG = 23          # HC-SR04 trigger
HC_ECHO = 24          # HC-SR04 echo
IR_PIN = 17           # E18-D80NK IR obstacle
TOUCH_PIN = 18        # TTP223 touch sensor

# I2C addresses
I2C_BUS = 1
MPU6050_ADDR = 0x68
BMP180_ADDR = 0x77
VL53L0X_ADDR = 0x29
GY302_ADDR = 0x23

# Global variables for sensors
vl53_sensor = None
vl53_type = None
dht_sensor = None

# =================== YOLO CONFIGURATION ===================

# Global variables for YOLO
yolo_net = None
yolo_output_layers = None
yolo_classes = None
yolo_loaded = False
yolo_active = False
yolo_lock = threading.Lock()

# YOLO Model paths (update these paths if needed)
YOLO_WEIGHTS = 'yolov4-tiny.weights'
YOLO_CONFIG = 'yolov4-tiny.cfg'
YOLO_CLASSES = 'coco.names'

# Get Raspberry Pi IP address
def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except:
        return "127.0.0.1"

# Global data storage
sensor_data = {
    'dht22': {'temp': deque(maxlen=100), 'humidity': deque(maxlen=100), 'latest': {}},  # Changed from dht11 to dht22
    'bmp180': {'pressure': deque(maxlen=100), 'latest': {}},
    'gy302': {'lux': deque(maxlen=100), 'latest': {}},
    'hcsr04': {'distance': deque(maxlen=100), 'latest': {}},
    'vl53l0x': {'distance': deque(maxlen=100), 'latest': {}},
    'ir_sensor': {'state': deque(maxlen=100), 'latest': {}},
    'touch_sensor': {'state': deque(maxlen=100), 'latest': {}},
    'mpu6050': {'accel_x': deque(maxlen=100), 'accel_y': deque(maxlen=100), 
                'accel_z': deque(maxlen=100), 'gyro_x': deque(maxlen=100),
                'gyro_y': deque(maxlen=100), 'gyro_z': deque(maxlen=100), 'latest': {}},
    'timestamps': deque(maxlen=100)
}
data_lock = threading.Lock()

# =================== YOLO FUNCTIONS ===================

def load_yolo_model():
    """Load YOLO model using OpenCV DNN"""
    global yolo_net, yolo_output_layers, yolo_classes, yolo_loaded
    
    try:
        print("Loading YOLO model...")
        
        # Load class names
        try:
            with open(YOLO_CLASSES, 'r') as f:
                yolo_classes = [line.strip() for line in f.readlines()]
        except:
            # Use default COCO classes if file not found
            yolo_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
                           'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
                           'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
                           'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
                           'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
                           'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
                           'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
                           'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
                           'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
                           'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 
                           'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
                           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
                           'scissors', 'teddy bear', 'hair drier', 'toothbrush']
            print(f"✓ Using default COCO classes ({len(yolo_classes)} classes)")
        
        # Load YOLO network
        yolo_net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
        
        # Use CPU (important for Raspberry Pi)
        yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get output layer names
        layer_names = yolo_net.getLayerNames()
        try:
            # OpenCV 4.x
            yolo_output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
        except:
            # OpenCV 3.x
            yolo_output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]
        
        # Test the model with a dummy image
        test_img = np.zeros((416, 416, 3), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(test_img, 1/255.0, (416, 416), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        _ = yolo_net.forward(yolo_output_layers)
        
        yolo_loaded = True
        print("✓ YOLO model loaded successfully")
        print(f"  Model: {YOLO_WEIGHTS}")
        print(f"  Classes loaded: {len(yolo_classes)}")
        return True
        
    except Exception as e:
        print(f"✗ Error loading YOLO model: {e}")
        print("Make sure you have downloaded the YOLO files:")
        print(f"  1. {YOLO_WEIGHTS} (download from https://github.com/AlexeyAB/darknet/releases)")
        print(f"  2. {YOLO_CONFIG} (from darknet/cfg folder)")
        print(f"  3. {YOLO_CLASSES} (from darknet/data folder)")
        print("\nOr run this command to download them:")
        print("wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights")
        print("wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg")
        print("wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names")
        
        yolo_loaded = False
        return False

def detect_objects_yolo(frame, confidence_threshold=0.4, nms_threshold=0.3):
    """Detect objects in frame using YOLO"""
    global yolo_net, yolo_output_layers, yolo_classes, yolo_loaded
    
    if not yolo_loaded:
        return [], 0, frame
    
    height, width = frame.shape[:2]
    
    # Prepare image for YOLO (resize to 416x416 for better performance)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    
    try:
        outputs = yolo_net.forward(yolo_output_layers)
    except Exception as e:
        print(f"YOLO detection error: {e}")
        return [], 0, frame
    
    # Process detections
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression to remove overlapping boxes
    if len(boxes) > 0:
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    else:
        indexes = []
    
    # Filter only "person" detections (class_id 0 in COCO dataset)
    person_boxes = []
    person_count = 0
    
    if len(indexes) > 0:
        # Handle different index formats
        if hasattr(indexes, 'shape'):
            # numpy array
            indexes = indexes.flatten()
        else:
            # single index or list
            indexes = [indexes] if not isinstance(indexes, list) else indexes
        
        for i in indexes:
            if i < len(class_ids) and class_ids[i] == 0:  # 0 = person in COCO dataset
                x, y, w, h = boxes[i]
                person_boxes.append({
                    'box': (x, y, w, h),
                    'confidence': confidences[i]
                })
                person_count += 1
    
    return person_boxes, person_count, frame

def draw_detections(frame, person_boxes, person_count):
    """Draw detection boxes and labels on frame"""
    # Draw person detections
    for detection in person_boxes:
        x, y, w, h = detection['box']
        confidence = detection['confidence']
        
        # Ensure coordinates are within frame bounds
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        # Draw bounding box
        color = (0, 255, 0)  # Green for persons
        thickness = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label with confidence
        label = f"Person: {confidence:.2f}"
        cv2.putText(frame, label, (x, max(y - 10, 20)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    # Add person count to top of frame
    cv2.putText(frame, f"Persons: {person_count}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

# =================== INITIALIZATION ===================

def setup():
    """Initialize hardware"""
    global vl53_sensor, vl53_type, dht_sensor
    
    try:
        bus = smbus2.SMBus(I2C_BUS)
        print("✓ I2C bus initialized")
    except:
        print("✗ I2C bus failed")
        sys.exit(1)
    
    # Setup GPIO
    GPIO.setup(HC_TRIG, GPIO.OUT)
    GPIO.setup(HC_ECHO, GPIO.IN)
    GPIO.setup(IR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(TOUCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.output(HC_TRIG, False)
    
    # Initialize VL53L0X
    vl53_sensor, vl53_type = initialize_vl53l0x()
    if vl53_sensor:
        print(f"✓ VL53L0X initialized (using {vl53_type} library)")
    else:
        print("✗ VL53L0X not initialized")
    
    # Initialize DHT22 (CHANGED FROM DHT11)
    try:
        import board
        import adafruit_dht
        dht_sensor = adafruit_dht.DHT22(board.D21)  # Changed to DHT22
        print("✓ DHT22 initialized")
    except ImportError:
        print("✗ adafruit_dht library not installed")
        print("  Install with: sudo pip3 install adafruit-circuitpython-dht")
    except Exception as e:
        print(f"✗ DHT22 initialization error: {e}")
    
    print("✓ GPIO initialized")
    return bus

def initialize_vl53l0x():
    """Initialize VL53L0X sensor"""
    try:
        import board
        import adafruit_vl53l0x
        i2c = board.I2C()
        vl53 = adafruit_vl53l0x.VL53L0X(i2c)
        return vl53, "adafruit"
    except ImportError:
        # Try old library
        try:
            import VL53L0X
            tof = VL53L0X.VL53L0X()
            tof.start_ranging(VL53L0X.VL53L0X_BETTER_ACCURACY_MODE)
            return tof, "vl53l0x"
        except ImportError:
            return None, None
    except Exception as e:
        return None, None

def read_vl53l0x_distance():
    """Read distance from VL53L0X sensor"""
    global vl53_sensor, vl53_type
    
    if vl53_sensor is None:
        return {'error': 'VL53L0X not initialized'}
    
    try:
        if vl53_type == "adafruit":
            distance = vl53_sensor.range
            if distance is not None:
                distance_cm = round(distance / 10.0, 1)
                # ADD OVER-RANGE CHECK (50cm limit)
                if distance_cm > 50:
                    return {
                        'distance_mm': distance,
                        'distance_cm': distance_cm,
                        'status': 'over ranged'  # This is the key line!
                    }
                else:
                    return {
                        'distance_mm': distance,
                        'distance_cm': distance_cm,
                        'status': 'OK'
                    }
            else:
                return {'error': 'No distance reading'}
                
        elif vl53_type == "vl53l0x":
            distance = vl53_sensor.get_distance()
            if distance > 0:
                distance_cm = round(distance / 10.0, 1)
                # ADD OVER-RANGE CHECK (50cm limit)
                if distance_cm > 50:
                    return {
                        'distance_mm': distance,
                        'distance_cm': distance_cm,
                        'status': 'over ranged'  # This is the key line!
                    }
                else:
                    return {
                        'distance_mm': distance,
                        'distance_cm': distance_cm,
                        'status': 'OK'
                    }
            else:
                return {'error': 'Invalid distance reading'}
                
    except Exception as e:
        return {'error': f'VL53L0X read error: {str(e)[:40]}'}
    
    return {'error': 'Unknown sensor type'}


# =================== SENSOR READING FUNCTIONS ===================

def read_dht22():  # Changed from read_dht11 to read_dht22
    """Read DHT22 temperature and humidity"""
    global dht_sensor
    
    if dht_sensor is None:
        return {'error': 'DHT22 not initialized'}
    
    try:
        temperature = dht_sensor.temperature
        humidity = dht_sensor.humidity
        
        if temperature is not None and humidity is not None:
            return {
                'temp_c': round(temperature, 1),  # DHT22 has higher precision
                'humidity': round(humidity, 1),   # DHT22 has higher precision
                'status': 'OK'
            }
        else:
            return {'error': 'No valid reading'}
    except RuntimeError as e:
        # DHT sensors can have occasional read errors
        return {'error': f'DHT22 read error: {str(e)[:30]}'}
    except Exception as e:
        return {'error': f'DHT22: {str(e)[:30]}'}

def read_ir_sensor():
    """Read IR obstacle sensor"""
    try:
        state = GPIO.input(IR_PIN)
        voltage = 3.3 if state == GPIO.HIGH else 0.0
        obstacle_detected = (state == GPIO.LOW)
        
        return {
            'obstacle': obstacle_detected,
            'state': state,
            'voltage': voltage,
            'status': 'OK'
        }
    except Exception as e:
        return {'error': f'IR Sensor: {str(e)[:30]}'}

def read_touch_sensor():
    """Read touch sensor"""
    try:
        state = GPIO.input(TOUCH_PIN)
        touched = (state == GPIO.HIGH)
        
        return {
            'touched': touched,
            'state': state,
            'status': 'OK'
        }
    except Exception as e:
        return {'error': f'Touch Sensor: {str(e)[:30]}'}

def read_hcsr04():
    """Read HC-SR04 ultrasonic sensor"""
    try:
        GPIO.output(HC_TRIG, True)
        time.sleep(0.00001)
        GPIO.output(HC_TRIG, False)
        
        timeout_start = time.time()
        while GPIO.input(HC_ECHO) == 0:
            if time.time() - timeout_start > 0.1:
                return {'error': 'Echo timeout'}
        
        pulse_start = time.time()
        while GPIO.input(HC_ECHO) == 1:
            pulse_end = time.time()
            if time.time() - pulse_start > 0.1:
                return {'error': 'Echo timeout'}
        
        pulse_duration = pulse_end - pulse_start
        distance_cm = (pulse_duration * 34300) / 2
        
        if 2 <= distance_cm <= 400:
            return {
                'distance_cm': round(distance_cm, 1),
                'status': 'OK'
            }
        else:
            return {
                'distance_cm': round(distance_cm, 1),
                'error': 'Out of range'
            }
    except Exception as e:
        return {'error': f'HC-SR04: {str(e)[:30]}'}

def read_gy302(bus):
    """Read GY302 light sensor"""
    try:
        bus.write_byte(GY302_ADDR, 0x20)
        time.sleep(0.18)
        data = bus.read_i2c_block_data(GY302_ADDR, 0, 2)
        lux = ((data[0] << 8) + data[1]) / 1.2
        
        if lux < 10:
            condition = "Very Dark"
        elif lux < 50:
            condition = "Dark"
        elif lux < 200:
            condition = "Dim"
        elif lux < 500:
            condition = "Normal"
        elif lux < 1000:
            condition = "Bright"
        elif lux < 10000:
            condition = "Very Bright"
        else:
            condition = "Direct Sun"
        
        return {
            'lux': round(lux, 1),
            'condition': condition,
            'status': 'OK'
        }
    except Exception as e:
        return {'error': f'GY302: {str(e)[:30]}'}

def read_mpu6050(bus):
    """Read MPU6050 accelerometer and gyroscope"""
    try:
        bus.write_byte_data(MPU6050_ADDR, 0x6B, 0)
        
        accel_x = bus.read_word_data(MPU6050_ADDR, 0x3B)
        accel_y = bus.read_word_data(MPU6050_ADDR, 0x3D)
        accel_z = bus.read_word_data(MPU6050_ADDR, 0x3F)
        
        gyro_x = bus.read_word_data(MPU6050_ADDR, 0x43)
        gyro_y = bus.read_word_data(MPU6050_ADDR, 0x45)
        gyro_z = bus.read_word_data(MPU6050_ADDR, 0x47)
        
        def to_signed(val):
            return val - 0x10000 if val >= 0x8000 else val
        
        accel_x = to_signed(accel_x) / 16384.0
        accel_y = to_signed(accel_y) / 16384.0
        accel_z = to_signed(accel_z) / 16384.0
        
        gyro_x = to_signed(gyro_x) / 131.0
        gyro_y = to_signed(gyro_y) / 131.0
        gyro_z = to_signed(gyro_z) / 131.0
        
        # Calculate roll and pitch from accelerometer
        roll = np.arctan2(accel_y, accel_z) * 180 / np.pi
        pitch = np.arctan2(-accel_x, np.sqrt(accel_y**2 + accel_z**2)) * 180 / np.pi
        
        return {
            'accel_x': round(accel_x, 3),
            'accel_y': round(accel_y, 3),
            'accel_z': round(accel_z, 3),
            'gyro_x': round(gyro_x, 1),
            'gyro_y': round(gyro_y, 1),
            'gyro_z': round(gyro_z, 1),
            'roll': round(roll, 1),
            'pitch': round(pitch, 1),
            'status': 'OK'
        }
    except Exception as e:
        return {'error': f'MPU6050: {str(e)[:30]}'}

def read_bmp180(bus):
    """Simplified BMP180 reading - often more reliable on Raspberry Pi"""
    try:
        # Try using a library if available
        try:
            import bmp180
            sensor = bmp180.BMP180(bus)
            pressure = sensor.get_pressure() / 100.0  # Convert Pa to hPa
            temp = sensor.get_temperature()
            return {
                'temp_c': round(temp, 1),
                'pressure_hpa': round(pressure, 1),
                'status': 'OK'
            }
        except ImportError:
            pass
        
        # Alternative simplified calculation
        bus.write_byte_data(BMP180_ADDR, 0xF4, 0x2E)  # Read temperature
        time.sleep(0.005)
        ut = bus.read_word_data(BMP180_ADDR, 0xF6)
        
        bus.write_byte_data(BMP180_ADDR, 0xF4, 0x34 + (3 << 6))  # Read pressure
        time.sleep(0.026)
        msb = bus.read_byte_data(BMP180_ADDR, 0xF6)
        lsb = bus.read_byte_data(BMP180_ADDR, 0xF7)
        xlsb = bus.read_byte_data(BMP180_ADDR, 0xF8)
        
        # Simplified calculation that often works
        raw_pressure = ((msb << 16) + (lsb << 8) + xlsb) >> (8 - 3)
        
        # Empirical adjustment - try different divisors if needed
        # Usually dividing by ~100 gives hPa, but you might need to adjust
        pressure_hpa = raw_pressure / 300.0
        
        # If still wrong, try one of these:
        # pressure_hpa = raw_pressure / 200.0  # If showing ~2000 hPa
        # pressure_hpa = raw_pressure / 300.0  # If showing ~3000 hPa
        
        # Temperature calculation (simplified)
        temperature = (ut - 0.0) / 10.0 - 50.0
        
        return {
            'temp_c': round(temperature, 1),
            'pressure_hpa': round(pressure_hpa, 1),
            'status': 'OK'
        }
    except Exception as e:
        return {'error': f'BMP180: {str(e)[:30]}'}
 

# =================== SENSOR READING THREAD ===================

def sensor_reading_thread(bus):
    """Background thread for reading sensors"""
    print("✓ Sensor reading thread started")
    reading_count = 0
    
    while True:
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            reading_count += 1
            
            # Read all sensors - Changed from read_dht11 to read_dht22
            readings = {
                'dht22': read_dht22(),  # Changed from 'dht11' to 'dht22'
                'bmp180': read_bmp180(bus),
                'gy302': read_gy302(bus),
                'hcsr04': read_hcsr04(),
                'ir_sensor': read_ir_sensor(),
                'touch_sensor': read_touch_sensor(),
                'mpu6050': read_mpu6050(bus),
                'vl53l0x': read_vl53l0x_distance()
            }
            
            # Store data
            with data_lock:
                sensor_data['timestamps'].append(timestamp)
                
                # Store each sensor's data
                for sensor_name, reading in readings.items():
                    if 'error' not in reading:
                        sensor_data[sensor_name]['latest'] = reading
                        # Store historical data
                        if sensor_name == 'dht22' and 'temp_c' in reading:  # Changed from 'dht11' to 'dht22'
                            sensor_data[sensor_name]['temp'].append(reading['temp_c'])
                            sensor_data[sensor_name]['humidity'].append(reading['humidity'])
                        elif sensor_name == 'bmp180' and 'pressure_hpa' in reading:
                            sensor_data[sensor_name]['pressure'].append(reading['pressure_hpa'])
                        elif sensor_name == 'gy302' and 'lux' in reading:
                            sensor_data[sensor_name]['lux'].append(reading['lux'])
                        elif sensor_name == 'hcsr04' and 'distance_cm' in reading:
                            sensor_data[sensor_name]['distance'].append(reading['distance_cm'])
                        elif sensor_name == 'ir_sensor' and 'state' in reading:
                            sensor_data[sensor_name]['state'].append(reading['state'])
                        elif sensor_name == 'touch_sensor' and 'state' in reading:
                            sensor_data[sensor_name]['state'].append(reading['state'])
                        elif sensor_name == 'mpu6050' and 'accel_x' in reading:
                            sensor_data[sensor_name]['accel_x'].append(reading['accel_x'])
                            sensor_data[sensor_name]['accel_y'].append(reading['accel_y'])
                            sensor_data[sensor_name]['accel_z'].append(reading['accel_z'])
                            sensor_data[sensor_name]['gyro_x'].append(reading['gyro_x'])
                            sensor_data[sensor_name]['gyro_y'].append(reading['gyro_y'])
                            sensor_data[sensor_name]['gyro_z'].append(reading['gyro_z'])
                        elif sensor_name == 'vl53l0x' and 'distance_cm' in reading:
                            sensor_data[sensor_name]['distance'].append(reading['distance_cm'])
            
            # Print to terminal every 20 readings
            if reading_count % 20 == 0:
                # Show VL53L0X status
                vlx_status = "Connected" if 'distance_cm' in readings['vl53l0x'] else "Error"
                print(f"[{timestamp}] Reading #{reading_count} | VL53L0X: {vlx_status}")
            
            time.sleep(0.5)  # Read every 500ms (0.5 seconds)
            
        except Exception as e:
            print(f"✗ Error in sensor thread: {e}")
            time.sleep(1)

# =================== CAMERA FUNCTIONS ===================

# Global camera variable
camera = None

def init_camera():
    """Initialize USB camera"""
    global camera
    try:
        # Try to open the USB camera (usually device 0 or 1)
        for camera_id in [0, 1, 2]:
            camera = cv2.VideoCapture(camera_id)
            if camera.isOpened():
                # Test if we can read a frame
                ret, test_frame = camera.read()
                if ret:
                    # Set camera properties for better performance
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640*2)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480*2)
                    camera.set(cv2.CAP_PROP_FPS, 18)  # Lower FPS for better performance
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
                    
                    print(f"✓ USB Camera initialized on device {camera_id}")
                    print(f"  Resolution: 640x480 at 20 FPS")
                    return True
                else:
                    camera.release()
        
        print("✗ No working USB camera found")
        return False
    except Exception as e:
        print(f"✗ Camera initialization error: {e}")
        return False

def generate_frames():
    """Generate camera frames for streaming with YOLO option"""
    global camera, yolo_active
    
    # For offline testing: create a dummy camera feed
    if camera is None or not camera.isOpened():
        # Create a test pattern
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "NO CAMERA CONNECTED", (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if yolo_active:
                cv2.putText(frame, "YOLO: READY (NO CAMERA)", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # If YOLO is active, perform detection
        if yolo_active and yolo_loaded:
            try:
                # Detect objects (only persons)
                person_boxes, person_count, processed_frame = detect_objects_yolo(
                    frame, 
                    confidence_threshold=0.4,  # Lower threshold for better detection
                    nms_threshold=0.3
                )
                
                # Draw detections
                frame = draw_detections(processed_frame, person_boxes, person_count)
                
                # Add YOLO status indicator
                cv2.putText(frame, f"YOLO ACTIVE - Persons: {person_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add FPS information (approximate)
                cv2.putText(frame, "Press 'YOLO' button to disable", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            except Exception as e:
                print(f"YOLO processing error: {e}")
                cv2.putText(frame, "YOLO ERROR", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        elif yolo_active and not yolo_loaded:
            # YOLO is active but model not loaded
            cv2.putText(frame, "YOLO MODEL NOT LOADED", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Resize if needed for consistent output
        if frame.shape[0] != 480 or frame.shape[1] != 640:
            frame = cv2.resize(frame, (640, 480))
        
        # Encode the frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Small delay to prevent overwhelming the Pi
        time.sleep(0.05)

# =================== WEB SERVER WITH FLASK ===================

from flask import Flask, render_template_string, jsonify, Response

app = Flask(__name__)

# HTML Template with Font Awesome for icons
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sensor Dashboard</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2ecc71;
            --danger-color: #e74c3c;
            --warning-color: #f39c12;
            --dark-color: #2c3e50;
            --light-color: #ecf0f1;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }
        
        .container {
            max-width: 1800px;
            margin: 0 auto;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .header-left {
            flex: 1;
            min-width: 400px;
        }
        
        .header-right {
            flex: 1;
            min-width: 500px;
            display: flex;
            justify-content: flex-end;
        }
        
        .header h1 {
            color: var(--dark-color);
            margin-bottom: 8px;
            font-size: 2.2em;
        }
        
        .header p {
            color: #7f8c8d;
            font-size: 0.95em;
            margin-bottom: 10px;
        }
        
        .status-bar {
            display: flex;
            gap: 10px;
            margin-top: 12px;
            flex-wrap: wrap;
        }
        
        .status-item {
            background: var(--primary-color);
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: 500;
        }
        
        /* Simplified Clock Styles - Larger and Cleaner */
        .simplified-clock {
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: white;
            padding: 25px 40px;
            border-radius: 15px;
            text-align: center;
            min-width: 500px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
            border: 3px solid rgba(46, 204, 113, 0.4);
        }
        
        .clock-time {
            font-size: 4em;
            font-weight: bold;
            color: #2ecc71;
            text-shadow: 0 0 20px rgba(46, 204, 113, 0.7);
            margin-bottom: 10px;
            font-family: 'Courier New', monospace;
            letter-spacing: 3px;
        }
        
        .clock-date {
            font-size: 1.8em;
            color: #ecf0f1;
            font-weight: 500;
            letter-spacing: 1px;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        @media (max-width: 1200px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            .header {
                flex-direction: column;
                gap: 20px;
            }
            .header-left, .header-right {
                width: 100%;
                min-width: auto;
            }
            .simplified-clock {
                min-width: auto;
                width: 100%;
            }
        }
        
        .left-column {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .right-column {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .section {
            background: rgba(255, 255, 255, 0.95);
            padding: 18px;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }
        
        .section-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--light-color);
        }
        
        .section-icon {
            font-size: 1.5em;
            margin-right: 10px;
            color: var(--primary-color);
        }
        
        .section-title {
            font-size: 1.2em;
            color: var(--dark-color);
        }
        
        .sensor-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }
        
        @media (max-width: 768px) {
            .sensor-grid {
                grid-template-columns: 1fr;
            }
            .clock-time {
                font-size: 3em;
            }
            .clock-date {
                font-size: 1.4em;
            }
        }
        
        .sensor-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .sensor-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
        }
        
        .sensor-header {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .sensor-icon {
            font-size: 1.2em;
            margin-right: 8px;
            width: 30px;
            height: 30px;
            background: var(--light-color);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .sensor-name {
            font-size: 0.95em;
            font-weight: 600;
            color: var(--dark-color);
        }
        
        .sensor-value {
            font-size: 1.6em;
            font-weight: 700;
            color: var(--primary-color);
            margin: 6px 0;
            text-align: center;
        }
        
        .sensor-unit {
            font-size: 0.7em;
            color: #7f8c8d;
            margin-left: 3px;
        }
        
        .sensor-status {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
            margin-top: 6px;
        }
        
        .status-ok {
            background: rgba(46, 204, 113, 0.1);
            color: var(--secondary-color);
        }
        
        .status-error {
            background: rgba(231, 76, 60, 0.1);
            color: var(--danger-color);
        }
        
        .status-warning {
            background: rgba(243, 156, 18, 0.1);
            color: var(--warning-color);
        }
        
        .accel-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 6px;
            margin-bottom: 10px;
        }
        
        .accel-item {
            text-align: center;
            padding: 6px;
            background: var(--light-color);
            border-radius: 5px;
        }
        
        .accel-label {
            font-size: 0.75em;
            color: #7f8c8d;
            margin-bottom: 3px;
        }
        
        .accel-value {
            font-size: 1em;
            font-weight: 600;
            color: var(--primary-color);
        }
        
        /* Camera feed container */
        .camera-container {
            width: 100%;
            height: 450px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: #2c3e50;
            border-radius: 12px;
            overflow: hidden;
            margin-top: 10px;
            position: relative;
        }
        
        .camera-feed {
            width: 100%;
            height: 100%;
            object-fit: cover;
            background: #1a252f;
        }
        
        .camera-controls {
            position: absolute;
            bottom: 15px;
            left: 0;
            right: 0;
            display: flex;
            justify-content: center;
            gap: 15px;
            z-index: 10;
        }
        
        .camera-btn {
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1em;
            transition: all 0.3s;
            font-weight: 500;
            min-width: 140px;
            justify-content: center;
        }
        
        .camera-btn:hover {
            background: rgba(52, 152, 219, 0.9);
            transform: translateY(-2px);
        }
        
        .camera-btn.yolo {
            background: rgba(231, 76, 60, 0.8);
        }
        
        .camera-btn.yolo:hover {
            background: rgba(231, 76, 60, 0.9);
        }
        
        .camera-btn.active {
            background: rgba(46, 204, 113, 0.9);
            box-shadow: 0 0 20px rgba(46, 204, 113, 0.5);
        }
        
        .camera-btn.active.yolo {
            background: rgba(46, 204, 113, 0.9);
        }
        
        .footer {
            text-align: center;
            color: white;
            margin-top: 20px;
            padding: 12px;
            font-size: 0.9em;
        }
        
        .update-info {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 8px;
            display: inline-block;
        }
        
        .timestamp {
            font-weight: 600;
            color: var(--warning-color);
        }
        
        .reading-number {
            font-weight: 600;
            color: var(--secondary-color);
        }
        
        /* Faster update indicator */
        .fast-update {
            animation: pulse 1s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        /* Camera status indicator */
        .camera-status {
            position: absolute;
            top: 15px;
            left: 15px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            display: flex;
            align-items: center;
            gap: 5px;
            z-index: 10;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #e74c3c;
        }
        
        .status-dot.active {
            background: #2ecc71;
            animation: blink 2s infinite;
        }
        
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Loading spinner for camera */
        .camera-loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            text-align: center;
            z-index: 5;
        }
        
        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top: 4px solid #3498db;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header with Split Layout -->
        <div class="header">
            <div class="header-left">
                <h1><i class="fas fa-microchip"></i> Raspberry Pi Sensor Dashboard</h1>
                <p>Real-time monitoring with <span class="fast-update" style="color: #2ecc71; font-weight: bold;">500ms updates</span></p>
                
                <div class="status-bar">
                    <div class="status-item"><i class="fas fa-sync-alt"></i> Update: <span id="refreshCount">0.5</span>s</div>
                    <div class="status-item"><i class="fas fa-clock"></i> Last: <span id="lastUpdate">--:--:--</span></div>
                    <div class="status-item"><i class="fas fa-chart-line"></i> Readings: <span id="totalReadings">0</span></div>
                </div>
            </div>
            
            <div class="header-right">
                <!-- Simplified Clock - Larger and Cleaner -->
                <div class="simplified-clock">
                    <div class="clock-time" id="simplifiedTime">--:--:--</div>
                    <div class="clock-date" id="simplifiedDate">-- --- --</div>
                </div>
            </div>
        </div>
        
        <div class="dashboard-grid">
            <!-- Left Column: Environmental Data & IMU -->
            <div class="left-column">
                <!-- Section 1: Environmental Data -->
                <div class="section">
                    <div class="section-header">
                        <div class="section-icon">
                            <i class="fas fa-cloud-sun"></i>
                        </div>
                        <div class="section-title">Environmental Data</div>
                    </div>
                    <div class="sensor-grid" id="environmentSection">
                        <!-- Data will be loaded by JavaScript -->
                        <p>Loading...</p>
                    </div>
                </div>
                
                <!-- Section 2: IMU & Distance -->
                <div class="section">
                    <div class="section-header">
                        <div class="section-icon">
                            <i class="fas fa-compass"></i>
                        </div>
                        <div class="section-title">IMU & Distance</div>
                    </div>
                    <div class="sensor-grid" id="imuSection">
                        <!-- Data will be loaded by JavaScript -->
                        <p>Loading...</p>
                    </div>
                </div>
            </div>
            
            <!-- Right Column: GPIO Sensors & Camera Feed -->
            <div class="right-column">
                <!-- Section 3: GPIO Sensors -->
                <div class="section">
                    <div class="section-header">
                        <div class="section-icon">
                            <i class="fas fa-plug"></i>
                        </div>
                        <div class="section-title">GPIO Sensors</div>
                    </div>
                    <div class="sensor-grid" id="gpioSection">
                        <!-- Data will be loaded by JavaScript -->
                        <p>Loading...</p>
                    </div>
                </div>
                
                <!-- Section 4: Camera Feed -->
                <div class="section">
                    <div class="section-header">
                        <div class="section-icon">
                            <i class="fas fa-camera"></i>
                        </div>
                        <div class="section-title">USB Camera Feed</div>
                    </div>
                    
                    <!-- Camera Feed Container -->
                    <div class="camera-container" id="cameraContainer">
                        <!-- Camera status indicator -->
                        <div class="camera-status">
                            <div class="status-dot" id="cameraStatusDot"></div>
                            <span id="cameraStatusText">Connecting...</span>
                        </div>
                        
                        <!-- Camera feed will be loaded here -->
                        <img class="camera-feed" id="cameraFeed" alt="USB Camera Feed">
                        
                        <!-- Camera controls -->
                        <div class="camera-controls">
                            <button class="camera-btn" id="pauseBtn" onclick="toggleCamera()">
                                <i class="fas fa-pause"></i> Pause
                            </button>
                            <button class="camera-btn" onclick="takeSnapshot()">
                                <i class="fas fa-camera"></i> Snapshot
                            </button>
                            <button class="camera-btn yolo" id="yoloBtn" onclick="toggleYOLO()">
                                <i class="fas fa-robot"></i> YOLO: <span id="yoloStatus">OFF</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <div class="update-info">
                <i class="fas fa-bolt"></i> 
                Dashboard updates every 500ms | 
                Reading #<span id="readingNumber">0</span>
            </div>
        </div>
    </div>
    
    <script>
        // Initialize variables
        let readingNumber = 0;
        let lastUpdateTime = Date.now();
        let cameraActive = true;
        let yoloActive = false;
        
        // Function to update the simplified clock
        function updateSimplifiedClock() {
            const now = new Date();
            
            // Update time
            const timeString = now.toLocaleTimeString('en-US', {
                hour12: false,
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
            document.getElementById('simplifiedTime').textContent = timeString;
            
            // Update date
            const dateString = now.toLocaleDateString('en-US', {
                weekday: 'long',
                year: 'numeric',
                month: 'long',
                day: 'numeric'
            });
            document.getElementById('simplifiedDate').textContent = dateString;
        }
        
        // Initialize camera feed on page load
        function initCameraFeed() {
            const cameraFeed = document.getElementById('cameraFeed');
            const statusDot = document.getElementById('cameraStatusDot');
            const statusText = document.getElementById('cameraStatusText');
            
            // Start with Flask video feed endpoint
            const cameraUrl = '/video_feed';
            
            cameraFeed.src = cameraUrl;
            cameraFeed.onload = function() {
                statusDot.classList.add('active');
                statusText.textContent = 'Live';
                console.log('Camera feed loaded successfully');
            };
            
            cameraFeed.onerror = function() {
                statusDot.classList.remove('active');
                statusText.textContent = 'Error';
                console.error('Failed to load camera feed');
            };
        }
        
        // Toggle camera pause/play
        function toggleCamera() {
            const btn = document.getElementById('pauseBtn');
            const feed = document.getElementById('cameraFeed');
            const statusText = document.getElementById('cameraStatusText');
            
            if (cameraActive) {
                // Pause
                btn.innerHTML = '<i class="fas fa-play"></i> Play';
                feed.style.opacity = '0.5';
                statusText.textContent = 'Paused';
                cameraActive = false;
                console.log('Camera paused');
            } else {
                // Play
                btn.innerHTML = '<i class="fas fa-pause"></i> Pause';
                feed.style.opacity = '1';
                statusText.textContent = 'Live';
                cameraActive = true;
                console.log('Camera playing');
            }
        }
        
        // Toggle YOLO detection
        function toggleYOLO() {
            const btn = document.getElementById('yoloBtn');
            const yoloStatus = document.getElementById('yoloStatus');
            
            yoloActive = !yoloActive;
            
            if (yoloActive) {
                btn.classList.add('active');
                yoloStatus.textContent = 'ON';
                console.log('YOLO detection enabled');
                
                // Send request to enable YOLO on server
                fetch('/toggle_yolo', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                }).then(response => response.json())
                  .then(data => {
                      console.log('YOLO toggle response:', data);
                  });
            } else {
                btn.classList.remove('active');
                yoloStatus.textContent = 'OFF';
                console.log('YOLO detection disabled');
                
                // Send request to disable YOLO on server
                fetch('/toggle_yolo', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                }).then(response => response.json())
                  .then(data => {
                      console.log('YOLO toggle response:', data);
                  });
            }
        }
        
        // Take snapshot
        function takeSnapshot() {
            const feed = document.getElementById('cameraFeed');
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas dimensions
            canvas.width = feed.videoWidth || 640;
            canvas.height = feed.videoHeight || 480;
            
            // Draw the current frame
            ctx.drawImage(feed, 0, 0, canvas.width, canvas.height);
            
            // Add timestamp
            ctx.fillStyle = 'white';
            ctx.font = '16px Arial';
            ctx.fillText(new Date().toLocaleString(), 10, 30);
            
            // Create download link
            const link = document.createElement('a');
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            link.download = 'snapshot-' + timestamp + '.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
            
            console.log('Snapshot taken and downloaded');
        }
        
        // Function to update all sensor data
        async function updateAllSensorData() {
            try {
                const startTime = Date.now();
                const response = await fetch('/api/data');
                const data = await response.json();
                
                // Calculate update speed
                const updateTime = Date.now() - startTime;
                const timeSinceLastUpdate = Date.now() - lastUpdateTime;
                lastUpdateTime = Date.now();
                
                // Update header information
                document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
                document.getElementById('totalReadings').textContent = data.total_readings || 0;
                readingNumber++;
                document.getElementById('readingNumber').textContent = readingNumber;
                
                // Update refresh count with actual update time
                document.getElementById('refreshCount').textContent = (timeSinceLastUpdate / 1000).toFixed(1);
                
                // ===== SECTION 1: Environmental Data =====
                let envHtml = '';
                
                // Temperature
                if (data.dht22.temp_c !== undefined && !data.dht22.error) {  // Changed from data.dht11 to data.dht22
                    envHtml += `
                    <div class="sensor-card">
                        <div class="sensor-header">
                            <div class="sensor-icon" style="color: #e74c3c;">
                                <i class="fas fa-thermometer-half"></i>
                            </div>
                            <div class="sensor-name">Temperature (DHT22)</div>  <!-- Added DHT22 label -->
                        </div>
                        <div class="sensor-value">
                            ${data.dht22.temp_c.toFixed(1)}<span class="sensor-unit">°C</span>
                        </div>
                        <div class="sensor-status status-ok">
                            <i class="fas fa-check-circle"></i> Normal
                        </div>
                    </div>
                    `;
                }
                
                // Humidity
                if (data.dht22.humidity !== undefined && !data.dht22.error) {  // Changed from data.dht11 to data.dht22
                    envHtml += `
                    <div class="sensor-card">
                        <div class="sensor-header">
                            <div class="sensor-icon" style="color: #3498db;">
                                <i class="fas fa-tint"></i>
                            </div>
                            <div class="sensor-name">Humidity (DHT22)</div>  <!-- Added DHT22 label -->
                        </div>
                        <div class="sensor-value">
                            ${data.dht22.humidity.toFixed(1)}<span class="sensor-unit">%</span>
                        </div>
                        <div class="sensor-status status-ok">
                            <i class="fas fa-check-circle"></i> Normal
                        </div>
                    </div>
                    `;
                }
                
                // Atmospheric Pressure
                if (data.bmp180.pressure_hpa !== undefined && !data.bmp180.error) {
                    envHtml += `
                    <div class="sensor-card">
                        <div class="sensor-header">
                            <div class="sensor-icon" style="color: #9b59b6;">
                                <i class="fas fa-weight-hanging"></i>
                            </div>
                            <div class="sensor-name">Atmospheric Pressure</div>
                        </div>
                        <div class="sensor-value">
                            ${data.bmp180.pressure_hpa.toFixed(1)}<span class="sensor-unit">hPa</span>
                        </div>
                        <div class="sensor-status status-ok">
                            <i class="fas fa-check-circle"></i> Normal
                        </div>
                    </div>
                    `;
                }
                
                // Light Intensity
                if (data.gy302.lux !== undefined && !data.gy302.error) {
                    let lightStatus = 'status-ok';
                    let lightIcon = 'check-circle';
                    let lightText = data.gy302.condition || 'Normal';
                    
                    if (data.gy302.lux < 10) {
                        lightStatus = 'status-warning';
                        lightIcon = 'moon';
                    } else if (data.gy302.lux < 50) {
                        lightStatus = 'status-warning';
                        lightIcon = 'cloud-moon';
                    } else if (data.gy302.lux > 1000) {
                        lightStatus = 'status-warning';
                        lightIcon = 'sun';
                    }
                    
                    envHtml += `
                    <div class="sensor-card">
                        <div class="sensor-header">
                            <div class="sensor-icon" style="color: #f39c12;">
                                <i class="fas fa-sun"></i>
                            </div>
                            <div class="sensor-name">Light Intensity</div>
                        </div>
                        <div class="sensor-value">
                            ${data.gy302.lux.toFixed(1)}<span class="sensor-unit">lux</span>
                        </div>
                        <div class="sensor-status ${lightStatus}">
                            <i class="fas fa-${lightIcon}"></i> ${lightText}
                        </div>
                    </div>
                    `;
                }
                
                // Update environment section
                document.getElementById('environmentSection').innerHTML = envHtml;
                
                // ===== SECTION 2: IMU & Distance =====
                let imuHtml = '';
                
                // Accelerometer
                if (data.mpu6050.accel_x !== undefined && !data.mpu6050.error) {
                    imuHtml += `
                    <div class="sensor-card">
                        <div class="sensor-header">
                            <div class="sensor-icon" style="color: #2ecc71;">
                                <i class="fas fa-bolt"></i>
                            </div>
                            <div class="sensor-name">Accelerometer</div>
                        </div>
                        <div class="accel-grid">
                            <div class="accel-item">
                                <div class="accel-label">X</div>
                                <div class="accel-value">${data.mpu6050.accel_x.toFixed(3)}</div>
                            </div>
                            <div class="accel-item">
                                <div class="accel-label">Y</div>
                                <div class="accel-value">${data.mpu6050.accel_y.toFixed(3)}</div>
                            </div>
                            <div class="accel-item">
                                <div class="accel-label">Z</div>
                                <div class="accel-value">${data.mpu6050.accel_z.toFixed(3)}</div>
                            </div>
                        </div>
                        <div class="sensor-status status-ok">
                            <i class="fas fa-check-circle"></i> Active
                        </div>
                    </div>
                    `;
                }
                
                // Gyroscope
                if (data.mpu6050.gyro_x !== undefined && !data.mpu6050.error) {
                    imuHtml += `
                    <div class="sensor-card">
                        <div class="sensor-header">
                            <div class="sensor-icon" style="color: #1abc9c;">
                                <i class="fas fa-sync-alt"></i>
                            </div>
                            <div class="sensor-name">Gyroscope</div>
                        </div>
                        <div class="accel-grid">
                            <div class="accel-item">
                                <div class="accel-label">X</div>
                                <div class="accel-value">${data.mpu6050.gyro_x.toFixed(1)}</div>
                            </div>
                            <div class="accel-item">
                                <div class="accel-label">Y</div>
                                <div class="accel-value">${data.mpu6050.gyro_y.toFixed(1)}</div>
                            </div>
                            <div class="accel-item">
                                <div class="accel-label">Z</div>
                                <div class="accel-value">${data.mpu6050.gyro_z.toFixed(1)}</div>
                            </div>
                        </div>
                        <div class="sensor-status status-ok">
                            <i class="fas fa-check-circle"></i> Active
                        </div>
                    </div>
                    `;
                }
                
                // Ultrasonic Distance
                if (data.hcsr04.distance_cm !== undefined && !data.hcsr04.error) {
                    let distanceStatus = 'status-ok';
                    let distanceIcon = 'check-circle';
                    let distanceText = 'Normal';
                    
                    if (data.hcsr04.distance_cm < 10) {
                        distanceStatus = 'status-warning';
                        distanceIcon = 'exclamation-triangle';
                        distanceText = 'Very Close';
                    } else if (data.hcsr04.distance_cm > 200) {
                        distanceStatus = 'status-warning';
                        distanceIcon = 'exclamation-triangle';
                        distanceText = 'Too Far';
                    }
                    
                    imuHtml += `
                    <div class="sensor-card">
                        <div class="sensor-header">
                            <div class="sensor-icon" style="color: #e67e22;">
                                <i class="fas fa-wave-square"></i>
                            </div>
                            <div class="sensor-name">Ultrasonic Distance</div>
                        </div>
                        <div class="sensor-value">
                            ${data.hcsr04.distance_cm.toFixed(1)}<span class="sensor-unit">cm</span>
                        </div>
                        <div class="sensor-status ${distanceStatus}">
                            <i class="fas fa-${distanceIcon}"></i> ${distanceText}
                        </div>
                    </div>
                    `;
                }
                
                // ToF Distance (VL53L0X) - WITH OVER RANGE DETECTION
if (data.vl53l0x.distance_cm !== undefined && !data.vl53l0x.error) {
    let tofStatus = 'status-ok';
    let tofIcon = 'check-circle';
    let tofText = 'Connected';
    let distanceDisplay = `${data.vl53l0x.distance_cm.toFixed(1)}<span class="sensor-unit">cm</span>`;
    let tofValueClass = '';
    
    // Check if ToF sensor status is 'over ranged'
    if (data.vl53l0x.status === 'over ranged') {
        tofStatus = 'status-overrange';
        tofIcon = 'exclamation-triangle';
        tofText = 'OVER RANGED (>50cm)';
        distanceDisplay = '>50<span class="sensor-unit">cm</span>';
        tofValueClass = 'tof-value-overrange';
    } else if (data.vl53l0x.distance_cm < 10) {
        tofStatus = 'status-warning';
        tofIcon = 'exclamation-triangle';
        tofText = 'Very Close';
    } else if (data.vl53l0x.distance_cm > 1000) {
        tofStatus = 'status-warning';
        tofIcon = 'exclamation-triangle';
        tofText = 'Too Far';
    }
    
    imuHtml += `
    <div class="sensor-card">
        <div class="sensor-header">
            <div class="sensor-icon ${data.vl53l0x.status === 'over ranged' ? 'tof-overrange' : ''}" style="color: ${data.vl53l0x.status === 'over ranged' ? '#e67e22' : '#34495e'};">
                <i class="fas fa-ruler"></i>
            </div>
            <div class="sensor-name">ToF Distance (VL53L0X)</div>
        </div>
        <div class="sensor-value ${tofValueClass}" style="${data.vl53l0x.status === 'over ranged' ? 'color: #e67e22;' : ''}">
            ${distanceDisplay}
        </div>
        <div class="sensor-status ${tofStatus}">
            <i class="fas fa-${tofIcon}"></i> ${tofText}
        </div>
    </div>
    `;
}
               
                
                // Update IMU section
                document.getElementById('imuSection').innerHTML = imuHtml;
                
                // ===== SECTION 3: GPIO Sensors =====
                let gpioHtml = '';
                
                // IR Sensor
                if (data.ir_sensor.state !== undefined && !data.ir_sensor.error) {
                    const irStatus = data.ir_sensor.obstacle ? 'status-error' : 'status-ok';
                    const irIcon = data.ir_sensor.obstacle ? 'exclamation-triangle' : 'check-circle';
                    const irText = data.ir_sensor.obstacle ? 'OBSTACLE DETECTED' : 'CLEAR PATH';
                    
                    gpioHtml += `
                    <div class="sensor-card">
                        <div class="sensor-header">
                            <div class="sensor-icon" style="color: ${data.ir_sensor.obstacle ? '#e74c3c' : '#2ecc71'};">
                                <i class="fas fa-exclamation-triangle"></i>
                            </div>
                            <div class="sensor-name">IR Obstacle Sensor</div>
                        </div>
                        <div class="sensor-value">
                            ${data.ir_sensor.voltage}<span class="sensor-unit">V</span>
                        </div>
                        <div class="sensor-status ${irStatus}">
                            <i class="fas fa-${irIcon}"></i> ${irText}
                        </div>
                    </div>
                    `;
                }
                
                // Touch Sensor
                if (data.touch_sensor.state !== undefined && !data.touch_sensor.error) {
                    const touchStatus = data.touch_sensor.touched ? 'status-ok' : '';
                    const touchIcon = data.touch_sensor.touched ? 'hand-paper' : 'hand-point-up';
                    const touchText = data.touch_sensor.touched ? 'TOUCHED' : 'READY';
                    
                    gpioHtml += `
                    <div class="sensor-card">
                        <div class="sensor-header">
                            <div class="sensor-icon" style="color: ${data.touch_sensor.touched ? '#9b59b6' : '#95a5a6'};">
                                <i class="fas fa-hand-paper"></i>
                            </div>
                            <div class="sensor-name">Touch Sensor</div>
                        </div>
                        <div class="sensor-value">
                            ${data.touch_sensor.touched ? 'ACTIVE' : 'IDLE'}
                        </div>
                        <div class="sensor-status ${touchStatus}">
                            <i class="fas fa-${touchIcon}"></i> ${touchText}
                        </div>
                    </div>
                    `;
                }
                
                // Update GPIO section
                document.getElementById('gpioSection').innerHTML = gpioHtml;
                
                // Show update time in console (for debugging)
                if (readingNumber % 20 === 0) {
                    console.log(`Update #${readingNumber}: ${updateTime}ms`);
                }
                
            } catch (error) {
                console.error('Error updating sensor data:', error);
            }
        }
        
        // Fast update - every 500ms
        let updateInterval = 500; // milliseconds
        
        // Initial load
        updateAllSensorData();
        updateSimplifiedClock();
        initCameraFeed(); // Start camera feed immediately
        
        // Update sensor data every 500ms
        setInterval(updateAllSensorData, updateInterval);
        
        // Update simplified clock every second (real-time)
        setInterval(updateSimplifiedClock, 1000);
        
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def get_sensor_data():
    """API endpoint for sensor data (JSON)"""
    with data_lock:
        # Get latest readings from all sensors - Changed from dht11 to dht22
        data = {
            'dht22': sensor_data['dht22']['latest'],  # Changed from 'dht11' to 'dht22'
            'bmp180': sensor_data['bmp180']['latest'],
            'gy302': sensor_data['gy302']['latest'],
            'hcsr04': sensor_data['hcsr04']['latest'],
            'ir_sensor': sensor_data['ir_sensor']['latest'],
            'touch_sensor': sensor_data['touch_sensor']['latest'],
            'mpu6050': sensor_data['mpu6050']['latest'],
            'vl53l0x': sensor_data['vl53l0x']['latest'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_readings': len(sensor_data['timestamps'])
        }
    return jsonify(data)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_yolo', methods=['POST'])
def toggle_yolo_route():
    """Toggle YOLO detection"""
    global yolo_active
    with yolo_lock:
        yolo_active = not yolo_active
        status = "enabled" if yolo_active else "disabled"
        print(f"YOLO detection {status}")
        return jsonify({
            'yolo_active': yolo_active,
            'message': f'YOLO detection {status}'
        })

@app.route('/api/status')
def get_status():
    """API endpoint for system status"""
    with data_lock:
        sensors_ok = 0
        total_sensors = 0
        sensor_list = []
        
        # Changed from 'dht11' to 'dht22'
        for sensor in ['dht22', 'bmp180', 'gy302', 'hcsr04', 'ir_sensor', 'touch_sensor', 'mpu6050', 'vl53l0x']:
            total_sensors += 1
            sensor_status = 'error'
            if 'status' in sensor_data[sensor]['latest'] and sensor_data[sensor]['latest']['status'] == 'OK':
                sensors_ok += 1
                sensor_status = 'ok'
            
            sensor_list.append({
                'name': sensor,
                'status': sensor_status,
                'last_update': len(sensor_data[sensor]['latest']) > 0
            })
        
        status = {
            'sensors_ok': sensors_ok,
            'total_sensors': total_sensors,
            'uptime': len(sensor_data['timestamps']),
            'status': 'online',
            'update_interval': '500ms',
            'camera_connected': camera is not None and camera.isOpened(),
            'yolo_active': yolo_active,
            'yolo_loaded': yolo_loaded,
            'sensors': sensor_list
        }
    return jsonify(status)

# =================== MAIN PROGRAM ===================

def main():
    """Main program"""
    print("="*60)
    print("SENSOR DASHBOARD - WITH USB CAMERA & YOLO DETECTION")
    print("DHT22 Temperature & Humidity Sensor")  # Updated text
    print("="*60)
    
    # Initialize hardware
    bus = setup()
    
    # Initialize camera
    init_camera()
    
    # Load YOLO model
    load_yolo_model()
    
    if yolo_loaded:
        print(f"\n🤖 YOLO Person Detection ready")
        print("   Model: yolov4-tiny")
        print("   Classes: 80 (COCO dataset)")
        print("   Person detection confidence: 40%")
    else:
        print(f"\n⚠️  YOLO not loaded - using camera feed only")
        print("   You can still use the camera without detection")
    
    # Start sensor reading thread
    sensor_thread = threading.Thread(
        target=sensor_reading_thread,
        args=(bus,),
        daemon=True
    )
    sensor_thread.start()
    
    # Get IP address
    ip_address = get_ip_address()
    print(f"\n🌐 Web Dashboard URLs:")
    print(f"   Local:  http://localhost:5000")
    print(f"   Network: http://{ip_address}:5000")
    print(f"\n📸 USB Camera feed automatically starts")
    print(f"   Camera stream: http://{ip_address}:5000/video_feed")
    print(f"\n🤖 YOLO Controls:")
    print("   - Click 'YOLO: OFF' button to enable detection")
    print("   - Green boxes will appear around detected persons")
    print("   - Person count shown at top of camera feed")
    print(f"\n📊 Sensor Configuration:")
    print("   • DHT22: High-precision temperature & humidity")  # Updated
    print("   • BMP180: Atmospheric pressure")
    print("   • GY302: Light intensity")
    print("   • HC-SR04: Ultrasonic distance")
    print("   • VL53L0X: Time-of-Flight distance")
    print("   • IR Sensor: Obstacle detection")
    print("   • Touch Sensor: Capacitive touch")
    print("   • MPU6050: Accelerometer & gyroscope")
    print(f"\n⚡ Sensor readings updating every 500ms...")
    print("   Press Ctrl+C to stop\n")
    
    try:
        # Start Flask web server
        try:
            from waitress import serve
            print("Starting web server on port 5000...")
            serve(app, host='0.0.0.0', port=5000)
        except ImportError:
            print("✗ Waitress not installed, using Flask development server")
            print("  Install with: pip install waitress")
            print("\nStarting Flask development server...")
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        
    finally:
        # Cleanup
        if 'dht_sensor' in globals() and dht_sensor:
            dht_sensor.exit()
        
        if vl53_type == "vl53l0x" and vl53_sensor:
            try:
                vl53_sensor.stop_ranging()
            except:
                pass
        
        if camera is not None:
            camera.release()
        
        GPIO.cleanup()
        bus.close()
        print("✓ Cleanup complete")
        print("="*60)

# =================== RUN SCRIPT ===================

if __name__ == "__main__":
    main()