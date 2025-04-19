import time
import datetime
import uuid
import json
import threading
import os
import RPi.GPIO as GPIO
import pigpio
import serial_for_url
from pathlib import Path
import signal
import sys


class HardwareSensor:
    """基础硬件传感器类"""

    def __init__(self, sensor_id=None, location="图书馆入口", interval=1.0):
        self.sensor_id = sensor_id or str(uuid.uuid4())[:8]
        self.location = location
        self.interval = interval
        self.running = False
        self.connection = None

    def connect(self):
        """连接到硬件传感器，子类实现具体逻辑"""
        raise NotImplementedError

    def disconnect(self):
        """断开与硬件传感器的连接，子类实现具体逻辑"""
        raise NotImplementedError

    def read_data(self):
        """从硬件读取传感器数据，子类实现具体逻辑"""
        raise NotImplementedError

    def get_single_reading(self):
        """获取单次传感器读数"""
        return self.read_data()

    def simulate(self, duration=60):
        """记录指定时长的传感器数据"""
        if not self.connection:
            self.connect()

        self.running = True
        data_points = []

        end_time = time.time() + duration

        while time.time() < end_time and self.running:
            try:
                data = self.read_data()
                data_points.append(data)
                time.sleep(min(self.interval, max(0, end_time - time.time())))
            except Exception as e:
                break

        self.running = False
        return data_points

    def stop(self):
        self.running = False
        self.disconnect()


class GPIOInfraredSensor(HardwareSensor):
    """红外传感器"""

    def __init__(self, pin_number=17, sensor_id=None, location="图书馆入口", interval=0.5):
        super().__init__(
            sensor_id or f"gpio_ir_{uuid.uuid4().hex[:8]}", location, interval)
        self.pin_number = pin_number
        self.cumulative_count = 0
        self.last_state = None
        self.gpio = None

    def connect(self):
        """连接到GPIO引脚"""
        try:
            self.gpio = GPIO

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin_number, GPIO.IN)

            self.last_state = GPIO.input(self.pin_number)
            self.connection = True
            return True
        except Exception as e:
            print(f"GPIO红外传感器配置失败: {str(e)}")
            self.connection = False
            return False

    def disconnect(self):
        if self.gpio and self.connection:
            try:
                self.gpio.cleanup(self.pin_number)
                self.connection = False
            except:
                pass

    def read_data(self):
        if not self.connection:
            self.connect()

        try:
            current_state = self.gpio.input(self.pin_number)

            if current_state != self.last_state:
                count = 1
                self.cumulative_count += count
                self.last_state = current_state

                return {
                    "sensor_id": self.sensor_id,
                    "type": "gpio_infrared",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "location": self.location,
                    "count": count,
                    "cumulative_count": self.cumulative_count,
                    "current_state": current_state,
                    "is_occupied": current_state == 0
                }
            else:
                return {
                    "sensor_id": self.sensor_id,
                    "type": "gpio_infrared",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "location": self.location,
                    "count": 0,
                    "cumulative_count": self.cumulative_count,
                    "current_state": current_state,
                    "is_occupied": current_state == 0
                }

        except Exception as e:
            print(f"读取GPIO红外传感器数据失败: {str(e)}")
            return None


class GPIOWiFiProbe(HardwareSensor):
    """WiFi探针"""

    def __init__(self, rx_pin=23, tx_pin=24, sensor_id=None, location="图书馆入口", interval=5.0):
        super().__init__(
            sensor_id or f"gpio_wifi_{uuid.uuid4().hex[:8]}", location, interval)
        self.rx_pin = rx_pin
        self.tx_pin = tx_pin
        self.gpio = None
        self.uart = None

    def connect(self):
        try:

            self.gpio = GPIO

            self.pi = pigpio.pi()
            if not self.pi.connected:
                print("无法连接到pigpio守护进程，请确保它正在运行")
                return False

            self.pi.set_mode(self.rx_pin, pigpio.INPUT)
            self.pi.set_mode(self.tx_pin, pigpio.OUTPUT)

            self.uart = serial_for_url(
                f"socket://localhost:{self.rx_pin},{self.tx_pin}?baud=9600")

            self.connection = True
            return True
        except Exception as e:
            print(f"GPIO WiFi探针配置失败: {str(e)}")
            self.connection = False
            return False

    def disconnect(self):
        """断开GPIO连接"""
        if self.connection:
            try:
                if self.uart:
                    self.uart.close()

                if hasattr(self, 'pi') and self.pi.connected:
                    self.pi.stop()

                self.connection = False
            except:
                pass

    def read_data(self):
        """读取WiFi探针数据"""
        if not self.connection:
            self.connect()

        try:
            self.uart.write(b'GET_STATS\r\n')

            time.sleep(0.5)
            response = b''
            while self.uart.in_waiting > 0:
                response += self.uart.read(self.uart.in_waiting)

            response_str = response.decode('utf-8').strip()

            if response_str:
                try:
                    data = json.loads(response_str)

                    active_count = data.get("active_devices", 0)
                    device_types = data.get("device_types", {})

                    return {
                        "sensor_id": self.sensor_id,
                        "type": "gpio_wifi_probe",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "location": self.location,
                        "active_devices_count": active_count,
                        "device_types": device_types
                    }
                except json.JSONDecodeError:
                    parts = response_str.split(',')
                    active_count = 0
                    device_types = {}

                    for part in parts:
                        if '=' in part:
                            key, value = part.split('=')
                            if key.strip() == 'count':
                                active_count = int(value.strip())
                            elif ':' in value:
                                device_type, count = value.strip().split(':')
                                device_types[device_type] = int(count)

                    return {
                        "sensor_id": self.sensor_id,
                        "type": "gpio_wifi_probe",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "location": self.location,
                        "active_devices_count": active_count,
                        "device_types": device_types
                    }
            else:
                print(f"WiFi探针没有返回数据")
                return None

        except Exception as e:
            print(f"读取GPIO WiFi探针数据失败: {str(e)}")
            return None


class SensorManager:
    """传感器管理类，负责管理多个传感器并处理数据"""

    def __init__(self, data_directory="sensor_data"):
        self.sensors = {}
        self.threads = {}
        self.data_directory = data_directory
        self.ensure_data_directory()
        self.running = False

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def ensure_data_directory(self):
        Path(self.data_directory).mkdir(parents=True, exist_ok=True)

    def add_sensor(self, sensor):
        self.sensors[sensor.sensor_id] = sensor
        return sensor.sensor_id

    def remove_sensor(self, sensor_id):
        if sensor_id in self.sensors:
            if sensor_id in self.threads and self.threads[sensor_id].is_alive():
                self.sensors[sensor_id].stop()
                self.threads[sensor_id].join(timeout=3.0)

            del self.sensors[sensor_id]
            if sensor_id in self.threads:
                del self.threads[sensor_id]
            return True
        return False

    def start_sensor(self, sensor_id):
        if sensor_id not in self.sensors:
            return False

        if sensor_id in self.threads and self.threads[sensor_id].is_alive():
            return False

        if not self.sensors[sensor_id].connect():
            return False

        thread = threading.Thread(
            target=self._sensor_loop,
            args=(sensor_id,),
            daemon=True
        )
        self.threads[sensor_id] = thread
        thread.start()
        return True

    def stop_sensor(self, sensor_id):
        if sensor_id in self.sensors:
            self.sensors[sensor_id].stop()
            if sensor_id in self.threads and self.threads[sensor_id].is_alive():
                self.threads[sensor_id].join(timeout=3.0)
            return True
        return False

    def start_all_sensors(self):
        self.running = True
        for sensor_id in self.sensors:
            self.start_sensor(sensor_id)

    def stop_all_sensors(self):
        self.running = False
        for sensor_id in list(self.sensors.keys()):
            self.stop_sensor(sensor_id)

    def _sensor_loop(self, sensor_id):
        """传感器数据采集循环"""
        sensor = self.sensors[sensor_id]

        file_path = os.path.join(
            self.data_directory,
            f"{sensor_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        try:
            with open(file_path, 'a') as f:
                while self.running:
                    data = sensor.get_single_reading()
                    if data:
                        f.write(json.dumps(data) + "\n")
                        f.flush()

                    else:
                        print(f"传感器 {sensor_id} 没有数据")

                    time.sleep(sensor.interval)
        except Exception as e:
            print(f"传感器 {sensor_id} 发生错误: {str(e)}")
        finally:
            sensor.disconnect()

    def signal_handler(self, sig, frame):
        """处理终止信号"""
        self.stop_all_sensors()
        sys.exit(0)


class SensorMonitor:

    def __init__(self, sensor_manager, check_interval=60):
        self.sensor_manager = sensor_manager
        self.check_interval = check_interval
        self.thread = None
        self.running = False

    def start(self):
        if self.thread and self.thread.is_alive():
            return False

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)

    def _monitor_loop(self):
        while self.running:
            for sensor_id, sensor in self.sensor_manager.sensors.items():
                thread_running = (
                    sensor_id in self.sensor_manager.threads and
                    self.sensor_manager.threads[sensor_id].is_alive()
                )

                if self.sensor_manager.running and not thread_running:
                    self.sensor_manager.stop_sensor(sensor_id)
                    time.sleep(2)  # 等待资源释放
                    self.sensor_manager.start_sensor(sensor_id)

            time.sleep(self.check_interval)


def main():
    """主函数"""
    manager = SensorManager()

    monitor = SensorMonitor(manager)

    config = {
        "gpio_infrared_sensors": [
            {"pin_number": 17, "location": "图书馆入口", "interval": 0.5},
            {"pin_number": 27, "location": "图书馆出口", "interval": 0.5},
            {"pin_number": 22, "location": "自习室", "interval": 0.5}
        ],
        "gpio_wifi_probes": [
            {"rx_pin": 23, "tx_pin": 24, "location": "图书馆入口", "interval": 5.0},
            {"rx_pin": 10, "tx_pin": 9, "location": "阅览室", "interval": 5.0}
        ]
    }

    try:
        for sensor_config in config.get("gpio_infrared_sensors", []):
            sensor = GPIOInfraredSensor(**sensor_config)
            manager.add_sensor(sensor)

        for sensor_config in config.get("gpio_wifi_probes", []):
            sensor = GPIOWiFiProbe(**sensor_config)
            manager.add_sensor(sensor)

        manager.start_all_sensors()

        monitor.start()

        print("GPIO传感器监测系统已启动...")

        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        print("用户中断，停止监测...")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        monitor.stop()
        manager.stop_all_sensors()


if __name__ == "__main__":
    main()
