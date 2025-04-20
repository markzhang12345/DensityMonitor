import numpy as np
import time
import datetime
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
        self.state = initial_state
        self.variance = initial_variance
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.min_variance_threshold = 1e-6

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
        if measurement is None:
            raise ValueError("测量值不能为空")

        R = measurement_variance if measurement_variance is not None else self.measurement_variance

        if hasattr(self, 'adaptive_noise') and self.adaptive_noise:
            innovation = measurement - self.state
            normalized_innovation = innovation**2 / (self.variance + R)
            if normalized_innovation > self.innovation_threshold:
                R = max(R, abs(innovation) * 0.5)

        if hasattr(self, 'state_transition_model') and self.state_transition_model is not None:
            predicted_state = self.state_transition_model(self.state)
        else:
            predicted_state = self.state

        if hasattr(self, 'control_input') and self.control_input is not None:
            if hasattr(self, 'control_model'):
                control_effect = self.control_model(self.control_input)
                predicted_state += control_effect

        if hasattr(self, 'dynamic_process_variance') and self.dynamic_process_variance:
            current_process_variance = self.calculate_process_variance()
        else:
            current_process_variance = self.process_variance

        predicted_variance = self.variance + current_process_variance

        if hasattr(self, 'forgetting_factor') and self.forgetting_factor < 1.0:
            predicted_variance /= self.forgetting_factor

        kalman_gain = predicted_variance / (predicted_variance + R)

        innovation = measurement - predicted_state

        if hasattr(self, 'max_gain') and kalman_gain > self.max_gain:
            kalman_gain = self.max_gain

        updated_state = predicted_state + kalman_gain * innovation

        if hasattr(self, 'state_bounds'):
            lower_bound, upper_bound = self.state_bounds
            updated_state = max(lower_bound, min(updated_state, upper_bound))

        updated_variance = (1 - kalman_gain) * predicted_variance

        if updated_variance < self.min_variance_threshold:
            updated_variance = self.min_variance_threshold

        if hasattr(self, 'track_metrics') and self.track_metrics:
            if not hasattr(self, 'innovation_history'):
                self.innovation_history = []
            self.innovation_history.append(innovation)

            if not hasattr(self, 'nis'):
                self.nis = innovation**2 / (predicted_variance + R)
            else:
                alpha = 0.95  # 平滑因子
                self.nis = alpha * self.nis + \
                    (1 - alpha) * (innovation**2 / (predicted_variance + R))

        self.state = updated_state
        self.variance = updated_variance

        return self.state, self.variance


class PeopleFlowEstimator:
    """人流密度估算器"""

    def __init__(self, area_size=100):
        """
        初始化人流密度估算器

        参数:
        area_size - 区域面积（平方米）
        """
        self.area_size = area_size

        self.ir_sensor = InfraredSensor(location="图书馆入口")
        self.wifi_sensor = WiFiProbe(location="图书馆入口")

        self.kalman_filter = KalmanFilter(
            initial_state=0,
            initial_variance=10,
            process_variance=2,
            measurement_variance=5
        )

        self.ir_weight = 0.6
        self.wifi_weight = 0.4

        self.history = []

    def _ir_count_to_density(self, ir_data):
        """将红外传感器数据转换为人流密度估计"""
        count = ir_data.get("count", 0)
        density = count / self.area_size
        variance = max(0.2 * count, 0.1)
        return density, variance

    def _wifi_count_to_density(self, wifi_data):
        """将WiFi探针数据转换为人流密度估计"""
        active_count = wifi_data.get("active_devices_count", 0)

        device_types = wifi_data.get("device_types", {})
        smartphones = device_types.get("smartphone", 0)
        laptops = device_types.get("laptop", 0)

        estimated_people = (smartphones / 1.2) + (laptops /
                                                  0.3) if smartphones > 0 or laptops > 0 else active_count / 1.5
        estimated_people = min(active_count, estimated_people)  # 取较小值作为保守估计

        density = estimated_people / self.area_size
        variance = max(0.5 * estimated_people, 0.1)

        return density, variance

    def _dynamic_sensor_fusion(self, ir_density, ir_variance, wifi_density, wifi_variance):
        """
        动态传感器融合
        基于当前测量的方差动态调整传感器权重
        """
        ir_variance = max(ir_variance, 1e-6)
        wifi_variance = max(wifi_variance, 1e-6)

        total_variance_inv = (1 / ir_variance) + (1 / wifi_variance)

        if total_variance_inv == 0:
            self.ir_weight = 0.6
            self.wifi_weight = 0.4
        else:
            self.ir_weight = (1 / ir_variance) / total_variance_inv
            self.wifi_weight = (1 / wifi_variance) / total_variance_inv

        fused_density = self.ir_weight * ir_density + self.wifi_weight * wifi_density
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
            ir_data = self.ir_sensor.get_single_reading()
            wifi_data = self.wifi_sensor.get_single_reading()

            ir_density, ir_variance = self._ir_count_to_density(ir_data)
            wifi_density, wifi_variance = self._wifi_count_to_density(
                wifi_data)

            fused_density, fused_variance = self._dynamic_sensor_fusion(
                ir_density, ir_variance, wifi_density, wifi_variance
            )

            filtered_density, filtered_variance = self.kalman_filter.update(
                measurement=fused_density,
                measurement_variance=fused_variance
            )

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

                if "error" in details:
                    print(f"错误: {details['error']}")
                    print("使用默认值继续...")
                else:
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


if __name__ == "__main__":
    try:
        estimator = PeopleFlowEstimator(area_size=200)

        print("开始人流密度估计...")
        print("区域: 图书馆入口, 面积: 200 平方米")
        print("=" * 50)

        results = estimator.continuous_estimation(duration=30, interval=2.0)

        if results:
            print("=" * 50)
            print(f"完成 {len(results)} 次人流密度估计")

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
