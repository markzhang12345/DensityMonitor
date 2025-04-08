import json
import datetime
import numpy as np
import os
from pathlib import Path
import random

# 创建数据目录
DATA_DIR = Path("./density_data")
DATA_DIR.mkdir(exist_ok=True)


def generate_daily_data(date, locations=None):
    """
    生成一天的人流密度数据

    参数:
    - date: 日期对象
    - locations: 位置列表，默认为None时使用默认位置列表

    返回:
    - 一天的人流密度数据列表
    """
    if locations is None:
        locations = ["商场入口", "中央广场", "地铁站", "餐饮区", "电影院", "图书馆入口"]

    data = []

    # 每个位置每天的基础密度模式（小时为单位）
    base_patterns = {
        "商场入口": [0.1, 0.05, 0.02, 0.01, 0.01, 0.02, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9,
                 1.0, 0.95, 0.9, 0.85, 0.9, 1.0, 0.95, 0.7, 0.5, 0.3, 0.2, 0.15],
        "中央广场": [0.05, 0.02, 0.01, 0.01, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9,
                 0.85, 0.8, 0.85, 0.9, 0.95, 0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1],
        "地铁站": [0.2, 0.1, 0.05, 0.02, 0.05, 0.3, 0.8, 1.2, 1.5, 1.0, 0.8, 0.9,
                1.0, 0.9, 0.8, 0.9, 1.3, 1.5, 1.2, 0.8, 0.6, 0.5, 0.4, 0.3],
        "餐饮区": [0.1, 0.05, 0.02, 0.01, 0.01, 0.05, 0.2, 0.4, 0.6, 0.5, 0.4, 0.9,
                1.2, 0.8, 0.5, 0.6, 0.8, 1.1, 1.0, 0.8, 0.5, 0.3, 0.2, 0.15],
        "电影院": [0.05, 0.02, 0.01, 0.01, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7,
                0.6, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1],
        "图书馆入口": [0.05, 0.02, 0.01, 0.01, 0.01, 0.05, 0.15, 0.3, 0.5, 0.7, 0.8, 0.75,
                  0.7, 0.75, 0.8, 0.85, 0.75, 0.65, 0.5, 0.4, 0.3, 0.25, 0.15, 0.1]
    }

    # 特殊日期模式（如节假日）
    special_dates = {
        (3, 8): "妇女节",
        (5, 1): "劳动节",
        (6, 1): "儿童节",
        (10, 1): "国庆节",
        (12, 25): "圣诞节",
        (4, 23): "世界读书日"  # 添加与图书馆相关的特殊日期
    }

    # 检查是否是特殊日期
    is_special_day = False
    special_day_factor = 1.0
    special_library_factor = 1.0
    date_key = (date.month, date.day)
    if date_key in special_dates:
        is_special_day = True
        special_day_factor = 1.5  # 节假日人流量增加50%
        # 世界读书日图书馆人流量特别增加
        if special_dates[date_key] == "世界读书日":
            special_library_factor = 2.0  # 世界读书日图书馆人流增加一倍

    # 期末考试季节（假设5月和12月是期末季）
    is_exam_period = date.month in [5, 12] and date.day > 15
    exam_factor = 1.3 if is_exam_period else 1.0  # 考试季图书馆人流增加30%

    # 每天每10分钟生成一条数据
    for location in locations:
        pattern = base_patterns.get(
            location, base_patterns["中央广场"])  # 默认使用中央广场的模式

        # 工作日和周末模式不同
        is_weekend = date.weekday() >= 5
        weekend_factor = 1.3 if is_weekend else 1.0

        # 图书馆在周末和工作日的模式可能不同（相对其他场所，周末可能人流反而少一些）
        if location == "图书馆入口":
            weekend_factor = 0.8 if is_weekend else 1.1  # 图书馆工作日更忙
            # 应用考试季因素
            weekend_factor *= exam_factor
            # 应用世界读书日因素
            if special_dates.get(date_key) == "世界读书日":
                weekend_factor *= special_library_factor

        # 综合因子：考虑周末和特殊日期
        combined_factor = weekend_factor * special_day_factor

        # 天气影响因子（随机模拟）
        weather_factor = np.random.choice(
            [0.7, 0.9, 1.0, 1.1], p=[0.1, 0.2, 0.5, 0.2])

        for hour in range(24):
            base_density = pattern[hour] * combined_factor * weather_factor

            # 每小时生成6条数据（10分钟间隔）
            for minute in range(0, 60, 10):
                timestamp = datetime.datetime(
                    date.year, date.month, date.day, hour, minute, 0
                )

                # 添加随机波动，模拟真实情况
                random_factor = np.random.normal(
                    1.0, 0.1)  # 均值为1.0，标准差为0.1的高斯分布
                density = max(0.01, base_density *
                              random_factor)  # 确保密度至少为0.01

                # 图书馆特殊事件（如讲座、展览等）
                if location == "图书馆入口" and random.random() < 0.01:  # 1%的概率有特殊活动
                    special_event_factor = random.uniform(1.5, 2.0)
                    density *= special_event_factor

                # 添加一些突发事件（低概率）
                if random.random() < 0.005:  # 0.5%概率
                    event_factor = np.random.choice(
                        [0.5, 1.5, 2.0])  # 人流突然减少或增加
                    density *= event_factor

                data.append({
                    "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                    "location": location,
                    "filtered_density": round(density, 2)
                })

    return data


def generate_dataset(start_date, days=30):
    """
    生成多天的数据集

    参数:
    - start_date: 起始日期
    - days: 天数

    返回:
    - 生成的文件路径列表
    """
    generated_files = []

    # 生成周期性趋势（每周的波动）
    weekly_trend = np.sin(np.linspace(0, 2*np.pi, 7)) * \
        0.15 + 1  # 生成-15%到+15%的周期性波动

    # 学期趋势（假设一学期约4个月，生成一个缓慢增长然后下降的趋势）
    semester_days = 120
    semester_phase = (np.arange(days) % semester_days) / semester_days
    semester_trend = np.sin(semester_phase * np.pi) * \
        0.2 + 0.9  # 学期初和学期末人少，学期中人多

    for day in range(days):
        current_date = start_date + datetime.timedelta(days=day)

        # 应用周期性趋势
        day_of_week = current_date.weekday()
        trend_factor = weekly_trend[day_of_week]

        # 学期趋势（主要影响图书馆）
        sem_trend = semester_trend[day % len(semester_trend)]

        # 生成该天的数据
        daily_data = generate_daily_data(current_date)

        # 应用周期性趋势到每个数据点（对图书馆特殊处理）
        for data_point in daily_data:
            if data_point["location"] == "图书馆入口":
                data_point["filtered_density"] = round(
                    data_point["filtered_density"] * trend_factor * sem_trend, 2)
            else:
                data_point["filtered_density"] = round(
                    data_point["filtered_density"] * trend_factor, 2)

        # 保存到JSON文件
        file_name = f"density_data_{current_date.strftime('%Y%m%d')}.json"
        file_path = DATA_DIR / file_name

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(daily_data, f, ensure_ascii=False, indent=2)

        generated_files.append(str(file_path))
        print(f"已生成 {file_path}，共 {len(daily_data)} 条数据")

    return generated_files


def generate_missing_data_scenario(files, missing_rate=0.05):
    """
    模拟数据缺失场景，随机删除一些数据点

    参数:
    - files: 文件路径列表
    - missing_rate: 数据缺失率
    """
    for file_path in random.sample(files, k=int(len(files) * 0.3)):  # 随机选择30%的文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 随机删除一些数据点
        indices_to_keep = random.sample(
            range(len(data)), k=int(len(data) * (1 - missing_rate)))
        filtered_data = [data[i] for i in sorted(indices_to_keep)]

        # 保存修改后的数据
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)

        print(f"已在 {file_path} 中模拟数据缺失，删除了 {len(data) - len(filtered_data)} 条数据")


def add_anomaly_data(files, anomaly_rate=0.02):
    """
    添加异常值数据

    参数:
    - files: 文件路径列表
    - anomaly_rate: 异常值比例
    """
    for file_path in random.sample(files, k=int(len(files) * 0.2)):  # 随机选择20%的文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 随机选择一些数据点添加异常值
        num_anomalies = int(len(data) * anomaly_rate)
        anomaly_indices = random.sample(range(len(data)), k=num_anomalies)

        for idx in anomaly_indices:
            # 异常值可能是异常高或异常低
            if random.random() < 0.5:
                # 异常高值
                data[idx]["filtered_density"] = round(
                    data[idx]["filtered_density"] * random.uniform(3.0, 5.0), 2)
            else:
                # 异常低值
                data[idx]["filtered_density"] = round(
                    data[idx]["filtered_density"] * random.uniform(0.05, 0.2), 2)

        # 保存修改后的数据
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"已在 {file_path} 中添加 {num_anomalies} 个异常值")


def generate_summary(files):
    """
    生成数据集摘要信息

    参数:
    - files: 文件路径列表
    """
    total_records = 0
    locations = set()
    min_density = float('inf')
    max_density = float('-inf')
    densities = []
    location_stats = {}

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_records += len(data)

        for record in data:
            location = record["location"]
            density = record["filtered_density"]

            locations.add(location)
            densities.append(density)
            min_density = min(min_density, density)
            max_density = max(max_density, density)

            # 收集每个位置的统计数据
            if location not in location_stats:
                location_stats[location] = {
                    "count": 0,
                    "sum": 0,
                    "min": float('inf'),
                    "max": float('-inf'),
                    "values": []
                }

            stats = location_stats[location]
            stats["count"] += 1
            stats["sum"] += density
            stats["min"] = min(stats["min"], density)
            stats["max"] = max(stats["max"], density)
            stats["values"].append(density)

    print("\n数据集摘要信息:")
    print(f"总文件数: {len(files)}")
    print(f"总记录数: {total_records}")
    print(f"位置数量: {len(locations)}")
    print(f"位置列表: {', '.join(sorted(locations))}")
    print(f"密度全局最小值: {min_density:.2f}")
    print(f"密度全局最大值: {max_density:.2f}")
    print(f"全局平均密度: {sum(densities)/len(densities):.2f}")
    print(f"全局密度标准差: {np.std(densities):.2f}")

    print("\n各位置统计信息:")
    for location, stats in sorted(location_stats.items()):
        avg = stats["sum"] / stats["count"]
        std = np.std(stats["values"])
        print(f"  {location}:")
        print(f"    数据点数量: {stats['count']}")
        print(f"    平均密度: {avg:.2f}")
        print(f"    密度标准差: {std:.2f}")
        print(f"    最小密度: {stats['min']:.2f}")
        print(f"    最大密度: {stats['max']:.2f}")


if __name__ == "__main__":
    print("开始生成30天的人流密度数据集（包含图书馆入口位置）...")

    # 生成从30天前到今天的数据
    end_date = datetime.datetime.now().date()
    start_date = end_date - datetime.timedelta(days=29)  # 总共30天（包括今天）

    print(f"生成从 {start_date} 到 {end_date} 的数据")

    # 生成基础数据集
    files = generate_dataset(start_date, days=30)
    print(f"已成功生成 {len(files)} 个数据文件的基础数据")

    # 模拟数据缺失
    generate_missing_data_scenario(files)

    # 添加异常值
    add_anomaly_data(files)

    # 生成摘要信息
    generate_summary(files)

    print("\n数据集生成完毕！数据存储在 ./density_data 目录下")
    print("您现在可以使用这些数据训练人流密度预测模型")
