import random
import time
import datetime
import uuid


class SensorSimulator:
    """基础传感器模拟器类"""

    def __init__(self, sensor_id=None, location="图书馆入口", interval=1.0):
        self.sensor_id = sensor_id or str(uuid.uuid4())[:8]
        self.location = location
        self.interval = interval
        self.running = False

    def generate_data(self):
        """生成传感器数据，子类实现具体逻辑"""
        raise NotImplementedError

    def get_single_reading(self):
        """获取单次传感器读数"""
        return self.generate_data()

    def simulate(self, duration=60):
        """模拟指定时长的传感器数据"""
        self.running = True
        data_points = []

        end_time = time.time() + duration

        while time.time() < end_time and self.running:
            data = self.generate_data()
            data_points.append(data)
            time.sleep(min(self.interval, end_time - time.time()))

        self.running = False
        return data_points

    def stop(self):
        """停止模拟"""
        self.running = False


class InfraredSensor(SensorSimulator):
    """红外传感器模拟器"""

    def __init__(self, sensor_id=None, location="图书馆入口"):
        super().__init__(
            sensor_id or f"ir_{uuid.uuid4().hex[:8]}", location, interval=1.0)
        self.cumulative_count = 0

    def generate_data(self):
        """生成简化的红外传感器数据"""
        # 简化的人流计数，随机生成0-30的数字
        count = random.randint(0, 30)
        self.cumulative_count += count

        # 简化的数据结构
        return {
            "sensor_id": self.sensor_id,
            "type": "infrared",
            "timestamp": datetime.datetime.now().isoformat(),  # 当前时间
            "location": self.location,  # 传感器位置
            "count": count,  # 当前读数
            "cumulative_count": self.cumulative_count,  # 累计读数
            "is_occupied": count > 0  # 是否有人经过
        }


class WiFiProbe(SensorSimulator):
    """WiFi探针模拟器"""

    def __init__(self, sensor_id=None, location="图书馆入口"):
        super().__init__(
            sensor_id or f"wifi_{uuid.uuid4().hex[:8]}", location, interval=5.0)
        # 设备类型
        self.device_types = ["smartphone", "laptop", "tablet", "iot_device"]

    def generate_data(self):
        """生成简化的WiFi探针数据"""
        # 随机生成当前活跃设备数量
        active_count = random.randint(5, 50)

        # 随机分配设备类型
        device_types_count = {}
        for _ in range(active_count):
            device_type = random.choices(
                self.device_types,
                weights=[0.7, 0.2, 0.08, 0.02]
            )[0]
            device_types_count[device_type] = device_types_count.get(
                device_type, 0) + 1

        # 简化的数据结构
        return {
            "sensor_id": self.sensor_id,
            "type": "wifi_probe",
            "timestamp": datetime.datetime.now().isoformat(),
            "location": self.location,
            "active_devices_count": active_count,  # 当前活跃设备数量
            "device_types": device_types_count  # 设备类型统计
        }


# 简单使用示例
def demo():
    """运行简单的演示"""
    # 创建传感器
    ir_sensor = InfraredSensor(location="图书馆入口")
    wifi_sensor = WiFiProbe(location="图书馆入口")

    # 获取单次读数
    print("红外传感器单次读数:")
    print(ir_sensor.get_single_reading())
    print("\nWiFi探针单次读数:")
    print(wifi_sensor.get_single_reading())

    # 模拟10秒的数据收集
    print("\n模拟红外传感器10秒数据:")
    ir_data = ir_sensor.simulate(10)
    print(f"收集到{len(ir_data)}条数据")
    print(f"最后一条数据: {ir_data[-1]}")

    print("\n模拟WiFi探针10秒数据:")
    wifi_data = wifi_sensor.simulate(10)
    print(f"收集到{len(wifi_data)}条数据")
    print(f"最后一条数据: {wifi_data[-1]}")


if __name__ == "__main__":
    demo()
