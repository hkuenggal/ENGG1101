#!/usr/bin/env python3
"""
Multi-Sensor Terminal Display
Shows real-time data from all 8 sensors in the terminal
"""

import time
import os
import sys
from smbus2 import SMBus
import RPi.GPIO as GPIO

# ============= Auto Elevate to Root =============

def run_with_sudo():
    """Re-run the script with sudo if not root"""
    if os.geteuid() != 0:
        print("⚠️  GPIO access requires root privileges")
        print("🔄 Re-running with sudo...")
        args = ['sudo', sys.executable] + sys.argv
        os.execvpe('sudo', args, os.environ)

# Automatically elevate if needed
run_with_sudo()

print("✅ Running with sufficient permissions\n")

# ============= Helper Function =============

def safe_format(value, default=0):
    """Safely format a value that might be None"""
    return value if value is not None else default

# ============= Sensor Classes =============

class BMP180:
    """BMP180 Temperature & Pressure Sensor"""
    def __init__(self, bus_num=1):
        try:
            self.bus = SMBus(bus_num)
            self.addr = 0x77
            self._read_calibration_data()
            self.temperature = 0
            self.pressure = 0
            self.enabled = True
            print("✅ BMP180 initialized")
        except Exception as e:
            print(f"⚠️  BMP180 init failed: {e}")
            self.enabled = False
        
    def _read_signed_16bit(self, reg):
        msb = self.bus.read_byte_data(self.addr, reg)
        lsb = self.bus.read_byte_data(self.addr, reg + 1)
        value = (msb << 8) + lsb
        return value - 65536 if value > 32767 else value
    
    def _read_unsigned_16bit(self, reg):
        msb = self.bus.read_byte_data(self.addr, reg)
        lsb = self.bus.read_byte_data(self.addr, reg + 1)
        return (msb << 8) + lsb
    
    def _read_calibration_data(self):
        self.AC1 = self._read_signed_16bit(0xAA)
        self.AC2 = self._read_signed_16bit(0xAC)
        self.AC3 = self._read_signed_16bit(0xAE)
        self.AC4 = self._read_unsigned_16bit(0xB0)
        self.AC5 = self._read_unsigned_16bit(0xB2)
        self.AC6 = self._read_unsigned_16bit(0xB4)
        self.B1 = self._read_signed_16bit(0xB6)
        self.B2 = self._read_signed_16bit(0xB8)
        self.MB = self._read_signed_16bit(0xBA)
        self.MC = self._read_signed_16bit(0xBC)
        self.MD = self._read_signed_16bit(0xBE)
    
    def read(self):
        if not self.enabled:
            return
        try:
            # Read temperature
            self.bus.write_byte_data(self.addr, 0xF4, 0x2E)
            time.sleep(0.005)
            msb = self.bus.read_byte_data(self.addr, 0xF6)
            lsb = self.bus.read_byte_data(self.addr, 0xF7)
            UT = (msb << 8) + lsb
            
            # Calculate temperature
            X1 = ((UT - self.AC6) * self.AC5) / 32768.0
            X2 = (self.MC * 2048.0) / (X1 + self.MD)
            B5 = X1 + X2
            self.temperature = (B5 + 8.0) / 16.0 / 10.0
            
            # Read pressure
            self.bus.write_byte_data(self.addr, 0xF4, 0x34)
            time.sleep(0.005)
            msb = self.bus.read_byte_data(self.addr, 0xF6)
            lsb = self.bus.read_byte_data(self.addr, 0xF7)
            xlsb = self.bus.read_byte_data(self.addr, 0xF8)
            UP = ((msb << 16) + (lsb << 8) + xlsb) >> 8
            
            # Calculate pressure
            B6 = B5 - 4000.0
            X1 = (self.B2 * (B6 * B6 / 4096.0)) / 2048.0
            X2 = (self.AC2 * B6) / 2048.0
            X3 = X1 + X2
            B3 = ((self.AC1 * 4.0 + X3) + 2.0) / 4.0
            X1 = (self.AC3 * B6) / 8192.0
            X2 = (self.B1 * (B6 * B6 / 4096.0)) / 65536.0
            X3 = (X1 + X2 + 2.0) / 4.0
            B4 = (self.AC4 * (X3 + 32768.0)) / 32768.0
            B7 = (UP - B3) * 50000.0
            p = (B7 * 2.0) / B4 if B7 < 0x80000000 else (B7 / B4) * 2.0
            X1 = (p / 256.0) ** 2
            X1 = (X1 * 3038.0) / 65536.0
            X2 = (-7357.0 * p) / 65536.0
            self.pressure = (p + (X1 + X2 + 3791.0) / 16.0) / 100  # hPa
        except Exception as e:
            pass

class DHT22Sensor:
    """DHT22 Temperature & Humidity Sensor"""
    def __init__(self):
        self.temperature = 0
        self.humidity = 0
        self.enabled = False
        try:
            import adafruit_dht
            import board
            self.sensor = adafruit_dht.DHT22(board.D21)
            self.method = "adafruit_dht"
            self.enabled = True
            print("✅ DHT22 initialized (adafruit_dht)")
        except:
            try:
                import Adafruit_DHT
                self.sensor = (Adafruit_DHT.DHT22, 21)
                self.method = "Adafruit_DHT"
                self.Adafruit_DHT = Adafruit_DHT
                self.enabled = True
                print("✅ DHT22 initialized (Adafruit_DHT)")
            except Exception as e:
                print(f"⚠️  DHT22 init failed: {e}")
    
    def read(self):
        if not self.enabled:
            return
        try:
            if self.method == "adafruit_dht":
                self.temperature = self.sensor.temperature
                self.humidity = self.sensor.humidity
            else:
                h, t = self.Adafruit_DHT.read(self.sensor[0], self.sensor[1])
                if h and t:
                    self.humidity = h
                    self.temperature = t
        except:
            pass

class GY302:
    """GY-302 Light Intensity Sensor"""
    def __init__(self, bus_num=1):
        try:
            self.bus = SMBus(bus_num)
            self.addr = 0x23
            self.light = 0
            self.enabled = True
            print("✅ GY-302 initialized")
        except Exception as e:
            print(f"⚠️  GY-302 init failed: {e}")
            self.enabled = False
    
    def read(self):
        if not self.enabled:
            return
        try:
            self.bus.write_byte(self.addr, 0x20)
            time.sleep(0.18)
            data = self.bus.read_i2c_block_data(self.addr, 0, 2)
            self.light = ((data[0] << 8) + data[1]) / 1.2
        except:
            pass

class MPU6050Sensor:
    """MPU6050 IMU Sensor - Using same logic as mpu_test.py"""
    def __init__(self, bus_num=1, address=0x68):
        self.enabled = False
        self.bus_num = bus_num
        self.address = address
        self.accel_x = 0
        self.accel_y = 0
        self.accel_z = 0
        self.gyro_x = 0
        self.gyro_y = 0
        self.gyro_z = 0
        
        try:
            # Initialize SMBus
            self.bus = SMBus(bus_num)
            
            # Check if device responds (read WHO_AM_I)
            who_am_i = self.bus.read_byte_data(self.address, 0x75)
            
            # Wake up MPU6050 (write 0 to power management register)
            self.bus.write_byte_data(self.address, 0x6B, 0x00)
            time.sleep(0.1)
            
            # Verify it woke up
            pwr_mgmt = self.bus.read_byte_data(self.address, 0x6B)
            
            self.enabled = True
            print("✅ MPU6050 initialized")
            
        except Exception as e:
            print(f"⚠️  MPU6050 init failed: {e}")
            self.enabled = False
    
    def read_word_2c(self, reg):
        """Read signed 16-bit value from two registers"""
        try:
            high = self.bus.read_byte_data(self.address, reg)
            low = self.bus.read_byte_data(self.address, reg + 1)
            val = (high << 8) + low
            if val >= 0x8000:
                return -((65535 - val) + 1)
            else:
                return val
        except:
            return 0
    
    def read(self):
        if not self.enabled:
            return
        try:
            # Read accelerometer data
            accel_x_raw = self.read_word_2c(0x3B)
            accel_y_raw = self.read_word_2c(0x3D)
            accel_z_raw = self.read_word_2c(0x3F)
            
            # Read gyroscope data
            gyro_x_raw = self.read_word_2c(0x43)
            gyro_y_raw = self.read_word_2c(0x45)
            gyro_z_raw = self.read_word_2c(0x47)
            
            # Convert to m/s² (accelerometer scale: ±2g)
            accel_scale = 16384.0
            accel_x_g = accel_x_raw / accel_scale
            accel_y_g = accel_y_raw / accel_scale
            accel_z_g = accel_z_raw / accel_scale
            
            self.accel_x = accel_x_g * 9.81
            self.accel_y = accel_y_g * 9.81
            self.accel_z = accel_z_g * 9.81
            
            # Convert to °/s (gyroscope scale: ±250°/s)
            gyro_scale = 131.0
            self.gyro_x = gyro_x_raw / gyro_scale
            self.gyro_y = gyro_y_raw / gyro_scale
            self.gyro_z = gyro_z_raw / gyro_scale
            
        except Exception as e:
            pass

class UltrasonicSensor:
    """HC-SR04 Ultrasonic Distance Sensor"""
    def __init__(self, trig=23, echo=24):
        try:
            self.trig = trig
            self.echo = echo
            GPIO.setup(self.trig, GPIO.OUT)
            GPIO.setup(self.echo, GPIO.IN)
            self.distance = 0
            self.enabled = True
            print("✅ Ultrasonic sensor initialized")
        except Exception as e:
            print(f"⚠️  Ultrasonic init failed: {e}")
            self.enabled = False
    
    def read(self):
        if not self.enabled:
            return
        try:
            GPIO.output(self.trig, False)
            time.sleep(0.002)
            GPIO.output(self.trig, True)
            time.sleep(0.00001)
            GPIO.output(self.trig, False)
            
            pulse_start = time.time()
            timeout_start = time.time()
            
            while GPIO.input(self.echo) == 0:
                pulse_start = time.time()
                if time.time() - timeout_start > 0.04:
                    return
            
            pulse_end = time.time()
            timeout_start = time.time()
            
            while GPIO.input(self.echo) == 1:
                pulse_end = time.time()
                if time.time() - timeout_start > 0.04:
                    return
            
            duration = pulse_end - pulse_start
            self.distance = (duration * 34300) / 2
            
            if self.distance > 400 or self.distance < 2:
                self.distance = 0
        except Exception as e:
            pass

class VL53L0XSensor:
    """VL53L0X TOF Distance Sensor"""
    def __init__(self):
        self.enabled = False
        try:
            import board
            import busio
            import adafruit_vl53l0x
            i2c = busio.I2C(board.SCL, board.SDA)
            self.sensor = adafruit_vl53l0x.VL53L0X(i2c)
            self.enabled = True
            print("✅ VL53L0X initialized")
        except Exception as e:
            print(f"⚠️  VL53L0X init failed: {e}")
            self.sensor = None
        self.distance = 0
    
    def read(self):
        if not self.enabled:
            return
        try:
            self.distance = self.sensor.range / 10  # Convert mm to cm
        except:
            pass

class IRSensor:
    """IR Obstacle Detection Sensor"""
    def __init__(self, pin=17):
        try:
            self.pin = pin
            GPIO.setup(self.pin, GPIO.IN)
            self.obstacle = False
            self.enabled = True
            print("✅ IR sensor initialized")
        except Exception as e:
            print(f"⚠️  IR Sensor init failed: {e}")
            self.enabled = False
    
    def read(self):
        if not self.enabled:
            return
        try:
            self.obstacle = (GPIO.input(self.pin) == GPIO.LOW)
        except:
            pass

class TouchSensor:
    """TTP223 Touch Sensor"""
    def __init__(self, pin=18):
        try:
            self.pin = pin
            GPIO.setup(self.pin, GPIO.IN)
            self.touched = False
            self.enabled = True
            print("✅ Touch sensor initialized")
        except Exception as e:
            print(f"⚠️  Touch Sensor init failed: {e}")
            self.enabled = False
    
    def read(self):
        if not self.enabled:
            return
        try:
            self.touched = GPIO.input(self.pin)
        except:
            pass

# ============= Main Display =============

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def print_header():
    """Print display header"""
    print("=" * 80)
    print(" " * 20 + "🔬 RASPBERRY PI - 8 SENSOR DASHBOARD")
    print("=" * 80)

def print_sensor_data(sensors):
    """Print all sensor data in organized format"""
    
    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│  📊 ENVIRONMENTAL SENSORS                                                   │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    # BMP180
    if sensors['bmp180'].enabled:
        temp = safe_format(sensors['bmp180'].temperature)
        pres = safe_format(sensors['bmp180'].pressure)
        print(f"  🌡️  BMP180 Temperature:  {temp:6.2f} °C")
        print(f"  📊 BMP180 Pressure:      {pres:6.2f} hPa")
    else:
        print("  ❌ BMP180: Not available")
    
    print()
    
    # DHT22
    if sensors['dht22'].enabled:
        temp = safe_format(sensors['dht22'].temperature)
        hum = safe_format(sensors['dht22'].humidity)
        print(f"  🌡️  DHT22 Temperature:   {temp:6.2f} °C")
        print(f"  💧 DHT22 Humidity:       {hum:6.2f} %")
    else:
        print("  ❌ DHT22: Not available")
    
    print()
    
    # GY-302
    if sensors['gy302'].enabled:
        light = safe_format(sensors['gy302'].light)
        print(f"  💡 Light Intensity:      {light:6.1f} lux")
    else:
        print("  ❌ GY-302: Not available")
    
    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│  📏 DISTANCE SENSORS                                                        │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    # Ultrasonic
    if sensors['ultrasonic'].enabled:
        dist = safe_format(sensors['ultrasonic'].distance)
        if dist > 0:
            print(f"  📡 HC-SR04 (Ultrasonic): {dist:6.2f} cm")
        else:
            print(f"  📡 HC-SR04 (Ultrasonic): Out of range")
    else:
        print("  ❌ HC-SR04: Not available")
    
    # VL53L0X
    if sensors['vl53l0x'].enabled:
        dist = safe_format(sensors['vl53l0x'].distance)
        print(f"  🎯 VL53L0X (TOF):        {dist:6.2f} cm")
    else:
        print("  ❌ VL53L0X: Not available")
    
    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│  🎮 MOTION SENSOR (MPU6050)                                                 │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    if sensors['mpu6050'].enabled:
        ax = safe_format(sensors['mpu6050'].accel_x)
        ay = safe_format(sensors['mpu6050'].accel_y)
        az = safe_format(sensors['mpu6050'].accel_z)
        gx = safe_format(sensors['mpu6050'].gyro_x)
        gy = safe_format(sensors['mpu6050'].gyro_y)
        gz = safe_format(sensors['mpu6050'].gyro_z)
        
        print(f"  🔴 Accelerometer X:      {ax:8.2f} m/s²")
        print(f"  🟢 Accelerometer Y:      {ay:8.2f} m/s²")
        print(f"  🔵 Accelerometer Z:      {az:8.2f} m/s²")
        print()
        print(f"  🔴 Gyroscope X:          {gx:8.2f} °/s")
        print(f"  🟢 Gyroscope Y:          {gy:8.2f} °/s")
        print(f"  🔵 Gyroscope Z:          {gz:8.2f} °/s")
    else:
        print("  ❌ MPU6050: Not available")
    
    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│  🔘 DIGITAL SENSORS                                                         │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    # IR Sensor
    if sensors['ir'].enabled:
        status = "🚨 OBSTACLE DETECTED!" if sensors['ir'].obstacle else "✅ Clear"
        print(f"  👁️  IR Sensor:            {status}")
    else:
        print("  ❌ IR Sensor: Not available")
    
    # Touch Sensor
    if sensors['touch'].enabled:
        status = "👆 TOUCHED!" if sensors['touch'].touched else "⭕ Not Touched"
        print(f"  ✋ Touch Sensor:          {status}")
    else:
        print("  ❌ Touch Sensor: Not available")
    
    print("\n" + "=" * 80)
    print("Press CTRL+C to exit")
    print("=" * 80)

def main():
    """Main function"""
    print("Initializing sensors...\n")
    
    # Clean up and set GPIO mode ONCE at the start
    GPIO.setwarnings(False)
    try:
        GPIO.cleanup()
    except:
        pass
    
    # Set GPIO mode globally (only once)
    GPIO.setmode(GPIO.BCM)
    print("✅ GPIO mode set to BCM\n")
    
    # Initialize all sensors in order
    sensors = {
        'bmp180': BMP180(),
        'dht22': DHT22Sensor(),
        'gy302': GY302(),
        'mpu6050': MPU6050Sensor(),
        'ultrasonic': UltrasonicSensor(),
        'vl53l0x': VL53L0XSensor(),
        'ir': IRSensor(),
        'touch': TouchSensor()
    }
    
    print("\n✅ All sensors initialization complete!\n")
    time.sleep(2)
    
    try:
        while True:
            # Read all sensors
            for sensor in sensors.values():
                sensor.read()
            
            # Clear screen and display
            clear_screen()
            print_header()
            print_sensor_data(sensors)
            
            # Update every 1 second
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        GPIO.cleanup()
        print("✅ Cleanup complete!")

if __name__ == "__main__":
    main()
