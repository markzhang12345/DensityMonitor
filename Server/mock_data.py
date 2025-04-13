
import json
import datetime
import random
import os
from pathlib import Path

OUTPUT_DIR = "./density_data"
DAYS_TO_GENERATE = 3
HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 6

LOCATIONS = [
    {"name": "商场入口", "area": 100.0, "base_density": 0.3,
        "day_peak": 16, "amplitude": 0.25},
    {"name": "中央广场", "area": 500.0, "base_density": 0.2,
        "day_peak": 14, "amplitude": 0.2},
    {"name": "地铁站", "area": 150.0, "base_density": 0.5,
        "day_peak": 8, "amplitude": 0.35},
    {"name": "餐饮区", "area": 300.0, "base_density": 0.25,
        "day_peak": 12, "amplitude": 0.3},
    {"name": "电影院", "area": 200.0, "base_density": 0.3,
        "day_peak": 20, "amplitude": 0.4},
    {"name": "图书馆入口", "area": 200.0, "base_density": 0.25,
        "day_peak": 11, "amplitude": 0.15}
]

Path(OUTPUT_DIR).mkdir(exist_ok=True)


def get_time_factor(hour, location):
    peak_hour = location["day_peak"]
    amplitude = location["amplitude"]

    hour_factor = amplitude * \
        (1 + 0.8 * (1 - abs((hour - peak_hour) % 24) / 12))

    if location["name"] == "餐饮区":
        # 餐饮区在早中晚有三个高峰
        meal_peaks = [7, 12, 18]
        meal_factor = 0
        for peak in meal_peaks:
            meal_factor += 0.8 * max(0, 1 - abs(hour - peak) / 2)
        hour_factor += meal_factor

    elif location["name"] == "地铁站":
        # 地铁站在早晚高峰人流密集
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            hour_factor *= 1.5

    elif location["name"] == "电影院":
        # 电影院白天人少，晚上人多
        if hour < 12:
            hour_factor *= 0.4
        elif 19 <= hour <= 22:
            hour_factor *= 1.3

    # 所有地点深夜人流减少
    if 23 <= hour or hour <= 5:
        hour_factor *= 0.2

    return hour_factor


def generate_density_data():
    """生成人流密度模拟数据"""
    all_records = []

    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=DAYS_TO_GENERATE)

    current_day = start_time
    while current_day < end_time:
        day_records = []

        for location in LOCATIONS:
            location_name = location["name"]
            area_size = location["area"]
            base_density = location["base_density"]

            weekday = current_day.weekday()
            weekend_factor = 1.3 if weekday >= 5 else 1.0

            for hour in range(24):
                # 计算该小时的基础人流密度（考虑时间因素）
                time_factor = get_time_factor(hour, location)
                hour_base_density = base_density * time_factor * weekend_factor

                for minute_interval in range(MINUTES_PER_HOUR):
                    record_time = current_day.replace(
                        hour=hour,
                        minute=minute_interval * (60 // MINUTES_PER_HOUR)
                    )

                    if record_time > end_time:
                        continue

                    random_factor = 1 + random.uniform(-0.1, 0.1)
                    filtered_density = max(
                        0.05, hour_base_density * random_factor)

                    ir_noise = random.uniform(-0.05, 0.08)
                    wifi_noise = random.uniform(-0.07, 0.05)
                    ir_density = max(0.02, filtered_density + ir_noise)
                    wifi_density = max(0.01, filtered_density + wifi_noise)

                    ir_count = int(ir_density * area_size)
                    wifi_count = int(wifi_density * area_size)
                    estimated_people = filtered_density * area_size

                    # 创建记录
                    record = {
                        "device_id": f"density_client_{location_name}_{record_time.strftime('%Y%m%d')}",
                        "timestamp": record_time.isoformat(),
                        "location": location_name,
                        "area_size": area_size,
                        "filtered_density": round(filtered_density, 3),
                        "ir_density": round(ir_density, 3),
                        "wifi_density": round(wifi_density, 3),
                        "ir_count": ir_count,
                        "wifi_count": wifi_count,
                        "estimated_people": round(estimated_people, 1),
                        "received_at": (record_time + datetime.timedelta(seconds=random.uniform(0.2, 3))).isoformat()
                    }

                    day_records.append(record)

        if day_records:
            date_str = current_day.strftime("%Y%m%d")
            output_file = os.path.join(
                OUTPUT_DIR, f"density_data_{date_str}.json")

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(day_records, f, ensure_ascii=False, indent=2)

            all_records.extend(day_records)
            print(f"已生成 {len(day_records)} 条记录保存到 {output_file}")

        # 移动到下一天
        current_day = current_day + datetime.timedelta(days=1)

    print(f"总共生成 {len(all_records)} 条记录")
    return all_records


if __name__ == "__main__":
    print("开始生成人流密度模拟数据...")
    generate_density_data()
    print("数据生成完成!")
