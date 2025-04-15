import time
import datetime
import uuid
import json
import logging
import threading
import os
from pathlib import Path
import signal
import sys
import serial
import requests


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


class InfraredSensor(HardwareSensor):
    """红外传感器硬件接口"""

    def __init__(self, device_path="/dev/ttyUSB0", sensor_id=None, location="图书馆入口",
                 baudrate=9600, timeout=1.0):
        super().__init__(
            sensor_id or f"ir_{uuid.uuid4().hex[:8]}", location, interval=1.0)
        self.device_path = device_path
        self.baudrate = baudrate
        self.timeout = timeout
        self.cumulative_count = 0

    def connect(self):
        """连接到红外传感器硬件"""
        try:
            self.connection = serial.Serial(
                port=self.device_path,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            return True
        except Exception as e:
            return False

    def disconnect(self):
        """断开红外传感器连接"""
        if self.connection and self.connection.is_open:
            self.connection.close()
            self.connection = None

    def read_data(self):
        """读取红外传感器数据"""
        if not self.connection or not self.connection.is_open:
            self.connect()

        try:
            # 发送读取命令
            self.connection.write(b'READ\r\n')
            # 等待响应
            response = self.connection.readline().decode('utf-8').strip()

            if response.startswith("COUNT:"):
                count = int(response.split(":")[1])
                self.cumulative_count += count

                return {
                    "sensor_id": self.sensor_id,
                    "type": "infrared",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "location": self.location,
                    "count": count,
                    "cumulative_count": self.cumulative_count,
                    "is_occupied": count > 0
                }
            else:
                return None

        except Exception as e:
            return None


class WiFiProbe(HardwareSensor):
    """WiFi探针硬件接口"""

    def __init__(self, ip_address="192.168.1.100", port=8080, sensor_id=None, location="图书馆入口"):
        super().__init__(
            sensor_id or f"wifi_{uuid.uuid4().hex[:8]}", location, interval=5.0)
        self.ip_address = ip_address
        self.port = port
        self.device_types = ["smartphone", "laptop", "tablet", "iot_device"]
        self.api_url = f"http://{ip_address}:{port}/api/stats"

    def connect(self):
        """连接到WiFi探针"""
        try:
            # 测试连接
            response = requests.get(
                f"http://{self.ip_address}:{self.port}/api/status",
                timeout=5
            )
            response.raise_for_status()
            self.connection = True
            return True
        except Exception as e:
            self.connection = False
            return False

    def disconnect(self):
        """断开WiFi探针连接"""
        self.connection = False

    def read_data(self):
        """读取WiFi探针数据"""
        if not self.connection:
            self.connect()

        try:
            response = requests.get(self.api_url, timeout=5)
            response.raise_for_status()
            data = response.json()

            active_count = data.get("active_devices", 0)
            device_types_count = data.get("device_types", {})

            return {
                "sensor_id": self.sensor_id,
                "type": "wifi_probe",
                "timestamp": datetime.datetime.now().isoformat(),
                "location": self.location,
                "active_devices_count": active_count,
                "device_types": device_types_count
            }

        except Exception as e:
            return None


class SensorManager:
    """传感器管理类，负责管理多个传感器并处理数据"""

    def __init__(self, data_directory="sensor_data"):
        self.sensors = {}
        self.threads = {}
        self.data_directory = data_directory
        self.ensure_data_directory()
        self.running = False

        # 注册信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def ensure_data_directory(self):
        """确保数据目录存在"""
        Path(self.data_directory).mkdir(parents=True, exist_ok=True)

    def add_sensor(self, sensor):
        """添加传感器到管理器"""
        self.sensors[sensor.sensor_id] = sensor
        return sensor.sensor_id

    def remove_sensor(self, sensor_id):
        """从管理器移除传感器"""
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
        """启动单个传感器数据采集"""
        if sensor_id not in self.sensors:
            return False

        if sensor_id in self.threads and self.threads[sensor_id].is_alive():
            return False

        # 尝试连接传感器
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
        """停止单个传感器数据采集"""
        if sensor_id in self.sensors:
            self.sensors[sensor_id].stop()
            if sensor_id in self.threads and self.threads[sensor_id].is_alive():
                self.threads[sensor_id].join(timeout=3.0)
            return True
        return False

    def start_all_sensors(self):
        """启动所有传感器数据采集"""
        self.running = True
        for sensor_id in self.sensors:
            self.start_sensor(sensor_id)

    def stop_all_sensors(self):
        """停止所有传感器数据采集"""
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
                        # 写入数据到文件
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


# 定义故障检测与恢复机制
class SensorMonitor:
    """传感器监控类，负责监控传感器状态并处理异常"""

    def __init__(self, sensor_manager, check_interval=60):
        self.sensor_manager = sensor_manager
        self.check_interval = check_interval
        self.thread = None
        self.running = False

    def start(self):
        """启动监控"""
        if self.thread and self.thread.is_alive():
            return False

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """停止监控"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)

    def _monitor_loop(self):
        """监控循环"""
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
        "infrared_sensors": [
            {"device_path": "/dev/ttyUSB0", "location": "图书馆入口", "baudrate": 9600},
            {"device_path": "/dev/ttyUSB1", "location": "图书馆出口", "baudrate": 9600},
            {"device_path": "/dev/ttyUSB2", "location": "自习室", "baudrate": 9600}
        ],
        "wifi_probes": [
            {"ip_address": "192.168.1.100", "port": 8080, "location": "图书馆入口"},
            {"ip_address": "192.168.1.101", "port": 8080, "location": "阅览室"}
        ]
    }

    try:
        for sensor_config in config["infrared_sensors"]:
            sensor = InfraredSensor(**sensor_config)
            manager.add_sensor(sensor)

        for sensor_config in config["wifi_probes"]:
            sensor = WiFiProbe(**sensor_config)
            manager.add_sensor(sensor)

        manager.start_all_sensors()

        monitor.start()

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
