from kalman_people_flow_density import PeopleFlowEstimator
from typing import Dict, Any
import datetime
import time
import argparse
import logging
import json
import requests
import sys

# 控制台和文件日志记录器配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("density_monitor.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DensityMonitor")


class DensityHttpClient:
    """人流密度HTTP客户端类，用于收集和发送人流密度数据到服务器"""
    # 构造函数

    def __init__(self, server_url="http://localhost:5000/api/density", area_size=200):
        """
        初始化人流密度HTTP客户端

        参数:
        server_url - 服务器URL，默认为本地5000端口
        area_size - 监测区域面积（平方米）
        """
        self.server_url = server_url
        # 初始化人流密度估计器
        self.estimator = PeopleFlowEstimator(area_size=area_size)
        # 生成唯一设备ID
        self.device_id = f"density_client_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    def _prepare_data(self, density_data: Dict[str, Any]) -> Dict[str, Any]:
        """准备要发送到服务器的数据"""
        return {
            "device_id": self.device_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "location": "图书馆入口",
            "area_size": density_data.get("area_size", 200),
            # 滤波后的密度数据
            "filtered_density": density_data.get("filtered_density", 0),
            # 红外线和WiFi密度数据
            "ir_density": density_data.get("ir_density", 0),
            "wifi_density": density_data.get("wifi_density", 0),

            "ir_count": density_data.get("ir_count", 0),
            "wifi_count": density_data.get("wifi_count", 0),
            "estimated_people": round(density_data.get("filtered_density", 0) * density_data.get("area_size", 200), 1)
        }

    def send_density(self) -> bool:
        """收集并发送当前人流密度"""
        try:
            # 使用卡尔曼滤波器估计人流密度
            density, details = self.estimator.estimate_density()

            # 打包数据
            data = self._prepare_data(details)
            json_data = json.dumps(data)

            # 打印将要发送的数据
            logger.info(
                f"估计人流密度: {density:.4f} 人/m², 预计人数: {data['estimated_people']}")

            # 发送数据
            logger.info(f"正在发送数据到 {self.server_url}...")
            response = requests.post(
                self.server_url,
                data=json_data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )

            # 检查响应
            if response.status_code in (200, 201):
                logger.info(f"数据发送成功!")
                return True
            else:
                logger.warning(
                    f"服务器返回错误代码: {response.status_code}, 响应: {response.text}")
                return False

        except requests.RequestException as e:
            logger.error(f"HTTP请求错误: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"发送数据时出错: {str(e)}")
            return False


def run_continuous_monitoring(server_url: str, area_size: float, interval: float, duration: float = None):
    """
    持续监测人流密度并发送到服务器

    参数:
    server_url - 服务器URL
    area_size - 区域面积（平方米）
    interval - 发送间隔（秒）
    duration - 监测持续时间（秒），None表示持续运行直到中断
    """
    # 创建客户端实例
    client = DensityHttpClient(server_url=server_url, area_size=area_size)

    logger.info(f"开始人流密度监测")
    logger.info(f"监测区域: 图书馆入口, 面积: {area_size} 平方米")
    logger.info(f"服务器地址: {server_url}")
    logger.info(f"发送间隔: {interval} 秒")

    if duration:
        logger.info(f"计划运行时长: {duration} 秒")
        end_time = time.time() + duration
    else:
        logger.info("将持续运行直到手动中断")
        end_time = float('inf')

    success_count = 0
    failure_count = 0

    try:
        while time.time() < end_time:
            start_process = time.time()

            # 发送数据
            if client.send_density():
                success_count += 1
            else:
                failure_count += 1

            # 计算等待时间
            process_time = time.time() - start_process
            wait_time = max(0, interval - process_time)

            if wait_time > 0:
                time.sleep(wait_time)

    except KeyboardInterrupt:
        logger.info("用户中断，停止监测...")
    finally:
        total = success_count + failure_count
        if total > 0:
            success_rate = (success_count / total) * 100
            logger.info(
                f"监测结束. 发送统计: 成功 {success_count}/{total} ({success_rate:.1f}%)")
        else:
            logger.info("监测结束. 未发送任何数据")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="人流密度监测系统")
    parser.add_argument("--server", type=str, default="http://localhost:5000/api/density",
                        help="服务器URL (默认: http://localhost:5000/api/density)")
    parser.add_argument("--area", type=float, default=200.0,
                        help="监测区域面积,平方米 (默认: 200)")
    parser.add_argument("--interval", type=float, default=10.0,
                        help="发送间隔,秒 (默认: 10)")
    parser.add_argument("--duration", type=float, default=None,
                        help="运行时长,秒 (默认: 无限)")
    parser.add_argument("--once", action="store_true",
                        help="只发送一次数据")

    args = parser.parse_args()

    try:
        if args.once:
            # 只发送一次
            client = DensityHttpClient(
                server_url=args.server, area_size=args.area)
            client.send_density()
        else:
            # 持续运行
            run_continuous_monitoring(
                server_url=args.server,
                area_size=args.area,
                interval=args.interval,
                duration=args.duration
            )
    except Exception as e:
        logger.error(f"程序运行时出错: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
