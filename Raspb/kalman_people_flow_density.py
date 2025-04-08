import numpy as np
import time
import datetime
from typing import Dict, List, Tuple
from simplified_sensor_simulator import InfraredSensor, WiFiProbe


class KalmanFilter:
    """卡尔曼滤波器实现"""

    def __init__(self, initial_state=0, initial_variance=1, process_variance=0.01, measurement_variance=0.1):
        """
        初始化卡尔曼滤波器

        参数:
        initial_state - 初始状态估计
        initial_variance - 初始估计误差协方差
        process_variance - 过程噪声协方差Q
        measurement_variance - 测量噪声协方差R
        """
        self.state = initial_state  # 初始状态估计x
        self.variance = initial_variance  # 初始估计误差协方差P
        self.process_variance = process_variance  # 过程噪声协方差Q
        self.measurement_variance = measurement_variance  # 测量噪声协方差R

    def update(self, measurement, measurement_variance=None):
        """
        卡尔曼滤波器更新

        参数:
        measurement - 当前测量值z
        measurement_variance - 当前测量的噪声协方差，如果提供则使用此值

        返回:
        updated_state - 更新后的状态估计
        updated_variance - 更新后的估计误差协方差
        """
        # 使用提供的测量噪声协方差或默认值
        R = measurement_variance if measurement_variance is not None else self.measurement_variance

        # 预测步骤
        predicted_state = self.state  # x' = x (简化模型，假设状态不变)
        predicted_variance = self.variance + self.process_variance  # P' = P + Q

        # 更新步骤
        kalman_gain = predicted_variance / \
            (predicted_variance + R)  # K = P' / (P' + R)
        self.state = predicted_state + kalman_gain * \
            (measurement - predicted_state)  # x = x' + K(z - x')
        self.variance = (1 - kalman_gain) * predicted_variance  # P = (1 - K)P'

        return self.state, self.variance


class PeopleFlowEstimator:
    """人流密度估算器"""

    def __init__(self, area_size=100):
        """
        初始化人流密度估算器

        参数:
        area_size - 区域面积（平方米）
        """
        self.area_size = area_size  # 区域面积（平方米）

        # 初始化传感器
        self.ir_sensor = InfraredSensor(location="图书馆入口")
        self.wifi_sensor = WiFiProbe(location="图书馆入口")

        # 初始化卡尔曼滤波器
        self.kalman_filter = KalmanFilter(
            initial_state=0,
            initial_variance=10,
            process_variance=2,
            measurement_variance=5
        )

        # 传感器权重系数（初始值）
        self.ir_weight = 0.6
        self.wifi_weight = 0.4

        # 记录历史数据
        self.history = []

    def _ir_count_to_density(self, ir_data):
        """将红外传感器数据转换为人流密度估计"""
        # 简单假设：红外传感器计数直接反映当前区域人数
        count = ir_data.get("count", 0)
        # 计算密度：人数/面积
        density = count / self.area_size
        # 估计方差（可基于经验设定）
        # 确保方差不为零，添加小的常数0.1
        variance = max(0.2 * count, 0.1)
        return density, variance

    def _wifi_count_to_density(self, wifi_data):
        """将WiFi探针数据转换为人流密度估计"""
        # WiFi探针计数需要考虑一人多设备的情况
        active_count = wifi_data.get("active_devices_count", 0)

        # 设备类型分布
        device_types = wifi_data.get("device_types", {})
        smartphones = device_types.get("smartphone", 0)
        laptops = device_types.get("laptop", 0)

        # 简单模型：估计实际人数
        # 假设平均每人携带1.2个智能手机，0.3个笔记本
        estimated_people = (smartphones / 1.2) + (laptops /
                                                  0.3) if smartphones > 0 or laptops > 0 else active_count / 1.5
        estimated_people = min(active_count, estimated_people)  # 取较小值作为保守估计

        # 计算密度
        density = estimated_people / self.area_size
        # 估计方差（WiFi数据通常比红外数据不稳定）
        # 确保方差不为零，添加小的常数0.1
        variance = max(0.5 * estimated_people, 0.1)

        return density, variance

    def _dynamic_sensor_fusion(self, ir_density, ir_variance, wifi_density, wifi_variance):
        """
        动态传感器融合
        基于当前测量的方差动态调整传感器权重
        """
        # 防止除以零错误，确保方差至少为一个很小的正数
        ir_variance = max(ir_variance, 1e-6)
        wifi_variance = max(wifi_variance, 1e-6)

        # 根据方差计算权重（方差越小，权重越大）
        total_variance_inv = (1 / ir_variance) + (1 / wifi_variance)

        # 如果总方差倒数为零（极不可能发生），使用默认权重
        if total_variance_inv == 0:
            self.ir_weight = 0.6
            self.wifi_weight = 0.4
        else:
            self.ir_weight = (1 / ir_variance) / total_variance_inv
            self.wifi_weight = (1 / wifi_variance) / total_variance_inv

        # 加权融合密度估计
        fused_density = self.ir_weight * ir_density + self.wifi_weight * wifi_density
        # 融合后的方差
        fused_variance = 1 / total_variance_inv if total_variance_inv > 0 else 1.0

        return fused_density, fused_variance

    def estimate_density(self):
        """
        估计当前人流密度

        返回：
        density - 估计的人流密度（人/平方米）
        details - 详细信息字典
        """
        try:
            # 获取传感器数据
            ir_data = self.ir_sensor.get_single_reading()
            wifi_data = self.wifi_sensor.get_single_reading()

            # 转换为密度估计
            ir_density, ir_variance = self._ir_count_to_density(ir_data)
            wifi_density, wifi_variance = self._wifi_count_to_density(
                wifi_data)

            # 动态传感器融合
            fused_density, fused_variance = self._dynamic_sensor_fusion(
                ir_density, ir_variance, wifi_density, wifi_variance
            )

            # 卡尔曼滤波更新
            filtered_density, filtered_variance = self.kalman_filter.update(
                measurement=fused_density,
                measurement_variance=fused_variance
            )

            # 记录历史数据
            timestamp = datetime.datetime.now().isoformat()
            history_entry = {
                "timestamp": timestamp,
                "ir_density": ir_density,
                "wifi_density": wifi_density,
                "fused_density": fused_density,
                "filtered_density": filtered_density,
                "ir_weight": self.ir_weight,
                "wifi_weight": self.wifi_weight
            }
            self.history.append(history_entry)

            # 返回估计结果和详细信息
            details = {
                "timestamp": timestamp,
                "ir_count": ir_data.get("count", 0),
                "wifi_count": wifi_data.get("active_devices_count", 0),
                "ir_density": ir_density,
                "wifi_density": wifi_density,
                "ir_weight": self.ir_weight,
                "wifi_weight": self.wifi_weight,
                "fused_density": fused_density,
                "filtered_density": filtered_density,
                "density_variance": filtered_variance,
                "area_size": self.area_size
            }

            return filtered_density, details
        except Exception as e:
            print(f"估计密度时出错: {str(e)}")
            # 返回安全的默认值
            timestamp = datetime.datetime.now().isoformat()
            return 0.0, {
                "timestamp": timestamp,
                "error": str(e),
                "ir_density": 0,
                "wifi_density": 0,
                "ir_weight": 0.5,
                "wifi_weight": 0.5,
                "fused_density": 0,
                "filtered_density": 0,
                "density_variance": 1.0,
                "area_size": self.area_size
            }

    def continuous_estimation(self, duration=60, interval=1.0):
        """
        连续估计指定时长的人流密度

        参数:
        duration - 估计持续时间（秒）
        interval - 估计间隔（秒）

        返回:
        results - 估计结果列表
        """
        results = []
        start_time = time.time()
        end_time = start_time + duration

        while time.time() < end_time:
            try:
                density, details = self.estimate_density()
                results.append(details)

                # 检查是否有错误
                if "error" in details:
                    print(f"错误: {details['error']}")
                    print("使用默认值继续...")
                else:
                    # 打印当前估计结果
                    print(f"时间: {details['timestamp']}")
                    print(
                        f"红外传感器: {details['ir_count']} 人, 密度: {details['ir_density']:.4f} 人/m²")
                    print(
                        f"WiFi探针: {details['wifi_count']} 设备, 密度: {details['wifi_density']:.4f} 人/m²")
                    print(
                        f"融合权重: 红外 {details['ir_weight']:.2f}, WiFi {details['wifi_weight']:.2f}")
                    print(f"融合密度: {details['fused_density']:.4f} 人/m²")
                    print(
                        f"滤波后密度: {details['filtered_density']:.4f} 人/m² (方差: {details['density_variance']:.4f})")
                    print("-" * 50)
            except Exception as e:
                print(f"连续估计过程中出错: {str(e)}")
                print("跳过当前时间步...")

            # 等待下一次估计
            remaining = min(interval, end_time - time.time())
            if remaining > 0:
                time.sleep(remaining)

        return results


# 主函数演示
if __name__ == "__main__":
    try:
        # 创建人流密度估算器（假设区域面积为200平方米）
        estimator = PeopleFlowEstimator(area_size=200)

        print("开始人流密度估计...")
        print("区域: 图书馆入口, 面积: 200 平方米")
        print("=" * 50)

        # 连续估计30秒
        results = estimator.continuous_estimation(duration=30, interval=2.0)

        if results:
            print("=" * 50)
            print(f"完成 {len(results)} 次人流密度估计")

            # 计算统计信息
            densities = [r.get('filtered_density', 0) for r in results]
            if densities:
                avg_density = np.mean(densities)
                max_density = np.max(densities)

                print(f"平均人流密度: {avg_density:.4f} 人/m²")
                print(f"最大人流密度: {max_density:.4f} 人/m²")
                print(f"估计区域内平均人数: {avg_density * 200:.1f} 人")
                print(f"估计区域内最大人数: {max_density * 200:.1f} 人")
            else:
                print("没有有效的密度估计结果")
        else:
            print("未收集到任何结果")
    except Exception as e:
        print(f"程序运行时出错: {str(e)}")
