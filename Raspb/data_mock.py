import requests
import time
import uuid
import random
import datetime
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SimpleSimulator")

DEFAULT_API_URL = "http://127.0.0.1:5000/api/density"

LOCATIONS = [
    {"name": "商场入口", "area_size": 100.0, "base_density": 0.3},
    {"name": "中央广场", "area_size": 500.0, "base_density": 0.2},
    {"name": "地铁站", "area_size": 150.0, "base_density": 0.5},
    {"name": "餐饮区", "area_size": 300.0, "base_density": 0.25},
    {"name": "电影院", "area_size": 200.0, "base_density": 0.3}
]


def generate_data(location):
    """生成单个位置的模拟数据"""
    now = datetime.datetime.now()

    base_density = location["base_density"]
    density_factor = 1.0 + (random.random() * 0.6 - 0.3)  # -30% 到 +30% 的随机波动

    hour = now.hour
    weekday = now.weekday()

    if weekday < 5:
        weekday_factor = 1.0
    else:
        weekday_factor = 1.5

    if 7 <= hour <= 9:
        hour_factor = 1.5
    elif 11 <= hour <= 13:
        hour_factor = 1.7
    elif 17 <= hour <= 19:
        hour_factor = 1.8
    elif 21 <= hour <= 23:
        hour_factor = 0.8
    elif 0 <= hour <= 6:
        hour_factor = 0.2
    else:
        hour_factor = 1.0

    # 餐饮区在午餐和晚餐时间人流增加
    if location["name"] == "餐饮区" and (11 <= hour <= 13 or 17 <= hour <= 19):
        hour_factor *= 1.5

    # 电影院在晚间人流增加
    if location["name"] == "电影院" and 19 <= hour <= 22:
        hour_factor *= 1.7

    filtered_density = base_density * density_factor * weekday_factor * hour_factor
    filtered_density = max(0.05, min(filtered_density, 3.0))

    # 计算人数
    estimated_people = filtered_density * location["area_size"]

    # 模拟传感器数据
    ir_density = filtered_density * (1 + random.uniform(-0.2, 0.2))
    wifi_density = filtered_density * (1 + random.uniform(-0.2, 0.2))
    ir_count = int(ir_density * location["area_size"])
    wifi_count = int(wifi_density * location["area_size"])

    # 生成设备ID (保持一致性)
    device_id = f"density_client_{location['name'].replace(' ', '_')}_{now.strftime('%Y%m%d')}"

    return {
        "device_id": device_id,
        "timestamp": now.isoformat(),
        "location": location["name"],
        "area_size": location["area_size"],
        "filtered_density": round(filtered_density, 3),
        "ir_density": round(ir_density, 3),
        "wifi_density": round(wifi_density, 3),
        "ir_count": ir_count,
        "wifi_count": wifi_count,
        "estimated_people": round(estimated_people, 1)
    }


def send_data(url, data):
    """发送数据到API"""
    try:
        response = requests.post(url, json=data)
        if response.status_code == 201:
            logger.info(
                f"成功发送数据: {data['location']} - 密度: {data['filtered_density']:.2f} - 人数: {data['estimated_people']:.1f}")
            return True
        else:
            logger.error(
                f"发送失败，状态码: {response.status_code}, 响应: {response.text}")
            return False
    except Exception as e:
        logger.error(f"请求异常: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="人流密度数据模拟器")
    parser.add_argument("--url", type=str,
                        default=DEFAULT_API_URL, help="API服务器地址")
    parser.add_argument("--interval", type=int, default=5, help="发送间隔(秒)")

    args = parser.parse_args()

    logger.info(f"开始向 {args.url} 发送模拟数据")
    logger.info(f"发送间隔: {args.interval}秒")
    logger.info("程序将持续运行，直到按Ctrl+C手动停止")

    sent_count = 0
    success_count = 0

    try:
        # 修改为无限循环，直到键盘中断
        while True:
            for location in LOCATIONS:
                data = generate_data(location)
                if send_data(args.url, data):
                    success_count += 1
                sent_count += 1

            if sent_count % 10 == 0:
                logger.info(f"已发送 {sent_count} 条数据，成功 {success_count} 条")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("收到中断信号，停止发送数据")

    logger.info(f"发送完成。总计: {sent_count}条, 成功: {success_count}条")


if __name__ == "__main__":
    main()
