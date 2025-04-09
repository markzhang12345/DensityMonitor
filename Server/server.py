from fastapi import FastAPI, Body, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import sys
import datetime
import json
import logging
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DensityMonitor")

# 定义数据模型


class DensityData(BaseModel):
    device_id: str = Field(..., description="设备唯一标识符")
    timestamp: str = Field(..., description="数据时间戳")
    location: str = Field(..., description="监测位置")
    area_size: float = Field(..., description="区域面积(平方米)")
    filtered_density: float = Field(..., description="滤波后人流密度(人/平方米)")
    ir_density: float = Field(0.0, description="红外估计人流密度")
    wifi_density: float = Field(0.0, description="WiFi估计人流密度")
    ir_count: Optional[int] = Field(None, description="红外计数")
    wifi_count: Optional[int] = Field(None, description="WiFi计数")
    estimated_people: float = Field(..., description="估计区域内人数")

    class Config:
        schema_extra = {
            "example": {
                "device_id": "density_client_20250323120000",
                "timestamp": "2025-03-23T12:00:00.000000",
                "location": "图书馆入口",
                "area_size": 200.0,
                "filtered_density": 0.25,
                "ir_density": 0.3,
                "wifi_density": 0.2,
                "ir_count": 60,
                "wifi_count": 45,
                "estimated_people": 50.0
            }
        }


class PredictionRequest(BaseModel):
    """预测请求模型"""
    location: Optional[str] = Field(None, description="预测位置，不指定则预测所有位置")
    hours: int = Field(24, description="预测未来小时数", ge=1, le=72)

    class Config:
        schema_extra = {
            "example": {
                "location": "图书馆入口",
                "hours": 24
            }
        }


# 创建数据存储目录
DATA_DIR = Path("./density_data")
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("./Model/models")
MODEL_DIR.mkdir(exist_ok=True)

# 内存中的数据存储
density_records = []
# LSTM预测模型类


class DensityPredictor:
    """人流密度预测模型类"""

    def __init__(self, model_path="./models/density_lstm_model.h5", sequence_length=24):
        """
        初始化预测器

        参数:
        - model_path: 模型文件路径
        - sequence_length: 用于预测的历史序列长度（小时）
        """
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # 尝试加载模型
        try:
            self.model = load_model(self.model_path)
            logger.info(f"成功加载模型: {self.model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            self.model = None

    def _load_data(self, location=None, days=3):
        """加载历史数据"""
        today = datetime.datetime.now()
        all_data = []

        # 计算起始日期
        start_date = today - datetime.timedelta(days=days)

        # 从内存中获取数据
        memory_data = [record for record in density_records if
                       (location is None or record["location"] == location)]

        # 从文件中获取数据
        data_files = sorted(DATA_DIR.glob("density_data_*.json"), reverse=True)
        for file_path in data_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)

                    # 过滤位置
                    if location:
                        file_data = [
                            r for r in file_data if r['location'] == location]

                    all_data.extend(file_data)
            except Exception as e:
                logger.warning(f"读取文件 {file_path} 时出错: {str(e)}")

        # 合并内存数据和文件数据
        all_data.extend(memory_data)

        if not all_data:
            logger.warning(f"没有找到符合条件的数据")
            return None

        # 转换为DataFrame
        import pandas as pd
        df = pd.DataFrame(all_data)

        # 确保时间戳格式正确并排序
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # 重采样到小时级别（取平均值）
        df.set_index('timestamp', inplace=True)
        hourly_df = df['filtered_density'].resample('h').mean()

        # 处理缺失值
        hourly_df = hourly_df.interpolate(method='linear')

        return hourly_df.reset_index()

    def predict_next_hours(self, location=None, hours_to_predict=24):
        """预测未来几小时的人流密度"""
        if self.model is None:
            logger.error("模型未加载，无法进行预测")
            return None

        # 加载最近数据
        df = self._load_data(location=location, days=3)
        if df is None or len(df) < self.sequence_length:
            logger.error("数据不足，无法进行预测")
            return None

        # 获取最后一个序列用于预测
        last_sequence = df['filtered_density'].values[-self.sequence_length:]
        scaled_sequence = self.scaler.fit_transform(
            df['filtered_density'].values.reshape(-1, 1))
        last_scaled_sequence = scaled_sequence[-self.sequence_length:]

        # 预测未来几小时
        current_sequence = last_scaled_sequence.reshape(
            1, self.sequence_length, 1)
        predictions = []
        timestamps = []

        # 获取最后一个时间戳
        last_timestamp = df['timestamp'].iloc[-1]

        for i in range(hours_to_predict):
            # 预测下一个值
            next_pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(next_pred[0, 0])

            # 更新时间戳
            next_timestamp = last_timestamp + datetime.timedelta(hours=i+1)
            timestamps.append(next_timestamp)

            # 更新序列（移除第一个值，添加预测值）
            current_sequence = np.append(current_sequence[:, 1:, :],
                                         next_pred.reshape(1, 1, 1),
                                         axis=1)

        # 转换回原始比例
        original_predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()

        # 组织返回结果
        result = {
            "location": location if location else "所有位置",
            "predictions": [
                {
                    "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                    "predicted_density": float(density),
                    "hour": timestamp.hour
                }
                for timestamp, density in zip(timestamps, original_predictions)
            ]
        }

        return result


# 全局预测器实例
predictor = None


def save_data_to_file(data_list):
    """将数据保存到JSON文件"""
    if not data_list:
        logger.info("没有数据需要保存")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    file_path = DATA_DIR / f"density_data_{timestamp}.json"

    try:
        # 读取现有数据(如果文件存在)
        existing_data = []
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"文件 {file_path} 格式错误，将使用空列表")
                existing_data = []

        # 合并数据并写入
        combined_data = existing_data + data_list

        # 创建临时文件，写入成功后再重命名，避免文件损坏
        temp_file = file_path.with_suffix('.tmp')
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)

        # 文件写入成功后移动到目标位置
        temp_file.replace(file_path)
        logger.info(f"成功保存 {len(data_list)} 条数据到 {file_path}")
    except Exception as e:
        logger.error(f"保存数据时出错: {str(e)}")
        raise

# 应用生命周期管理


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用启动时执行
    logger.info("应用服务启动...")
    # 初始化预测器
    global predictor
    try:
        predictor = DensityPredictor(model_path=str(
            MODEL_DIR / "density_lstm_model.h5"))
        if predictor.model is None:
            logger.warning("预测模型未能正确加载，预测功能将不可用")
        else:
            logger.info("预测模型已成功加载")
    except Exception as e:
        logger.error(f"初始化预测器时出错: {str(e)}")
        predictor = None

    yield  # 应用运行期间

    # 应用关闭时执行
    logger.info("应用关闭中...")
    if density_records:
        try:
            save_data_to_file(density_records)
            logger.info(f"应用关闭，已保存 {len(density_records)} 条数据")
        except Exception as e:
            logger.error(f"应用关闭时保存数据失败: {str(e)}")


# 创建FastAPI应用
app = FastAPI(
    title="人流密度监测服务",
    description="接收、存储和查询人流密度数据的API服务",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# API路由


@app.get("/")
async def root():
    """API根路径，返回基本信息"""
    return {
        "message": "人流密度监测与预测服务API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/density": "提交人流密度数据",
            "GET /api/density/recent": "获取最近的人流密度数据",
            "GET /api/density/statistics": "获取人流密度统计信息",
            "POST /api/density/predict": "预测未来人流密度"
        }
    }


@app.post("/api/density", status_code=201)
async def add_density_data(data: DensityData = Body(...)):
    """接收人流密度数据"""
    try:
        # 添加接收时间戳
        record = data.model_dump()
        record["received_at"] = datetime.datetime.now().isoformat()

        # 存储到内存
        density_records.append(record)

        # 如果记录超过100条，保存到文件并清空内存
        if len(density_records) >= 100:
            records_to_save = density_records.copy()
            density_records.clear()
            try:
                save_data_to_file(records_to_save)
            except Exception as e:
                # 如果保存失败，恢复内存中的记录
                density_records.extend(records_to_save)
                logger.error(f"保存数据失败: {str(e)}")
                # 不向客户端抛出异常，继续处理请求

        logger.info(f"接收到数据: {record['device_id']} - {record['location']}")

        return {
            "status": "success",
            "message": "数据已接收",
            "received_at": record["received_at"]
        }
    except Exception as e:
        logger.error(f"处理数据请求时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器处理错误: {str(e)}")


@app.get("/api/density/recent")
async def get_recent_data(limit: int = Query(10, description="返回记录数量", ge=1, le=100)):
    """获取最近的人流密度数据"""
    try:
        # 从内存中获取最近记录
        recent_records = density_records[-limit:] if density_records else []

        # 如果内存中记录不足，则从最新的文件中读取
        if len(recent_records) < limit:
            needed = limit - len(recent_records)
            try:
                # 获取最新的数据文件
                data_files = sorted(DATA_DIR.glob(
                    "density_data_*.json"), reverse=True)
                if data_files:
                    with open(data_files[0], "r", encoding="utf-8") as f:
                        file_records = json.load(f)
                        # 添加来自文件的记录
                        file_records = file_records[-needed:] if len(
                            file_records) > needed else file_records
                        # 合并记录(文件记录在前，因为它们更早)
                        recent_records = file_records + recent_records
            except Exception as e:
                # 如果读取文件失败，只返回内存中的记录
                logger.error(f"读取数据文件时出错: {str(e)}")

        return {
            "count": len(recent_records),
            "data": recent_records
        }
    except Exception as e:
        logger.error(f"获取最近数据时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器处理错误: {str(e)}")


@app.get("/api/density/statistics")
async def get_statistics(location: Optional[str] = None, hours: int = Query(24, description="统计时间范围(小时)", ge=1, le=168)):
    """获取人流密度统计信息"""
    try:
        # 计算时间范围
        now = datetime.datetime.now()
        start_time = (now - datetime.timedelta(hours=hours)).isoformat()

        # 收集所有记录
        all_records = density_records.copy()

        # 从文件中读取数据
        try:
            data_files = sorted(DATA_DIR.glob(
                "density_data_*.json"), reverse=True)
            for file_path in data_files:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_records = json.load(f)
                    all_records.extend(file_records)
        except Exception as e:
            logger.error(f"读取统计数据时出错: {str(e)}")

        # 过滤数据
        filtered_records = []
        for record in all_records:
            # 过滤时间范围
            if record["timestamp"] >= start_time:
                # 如果指定了位置，过滤位置
                if location is None or record["location"] == location:
                    filtered_records.append(record)

        # 如果没有数据，返回空统计
        if not filtered_records:
            return {
                "status": "success",
                "message": "没有符合条件的数据",
                "statistics": {
                    "count": 0,
                    "period": f"过去{hours}小时",
                    "location": location if location else "所有位置"
                }
            }

        # 计算统计数据
        densities = [r["filtered_density"] for r in filtered_records]
        people_counts = [r["estimated_people"] for r in filtered_records]

        statistics = {
            "count": len(filtered_records),
            "period": f"过去{hours}小时",
            "location": location if location else "所有位置",
            "density": {
                "current": densities[-1] if densities else 0,
                "average": sum(densities) / len(densities) if densities else 0,
                "max": max(densities) if densities else 0,
                "min": min(densities) if densities else 0
            },
            "people": {
                "current": people_counts[-1] if people_counts else 0,
                "average": sum(people_counts) / len(people_counts) if people_counts else 0,
                "max": max(people_counts) if people_counts else 0,
                "min": min(people_counts) if people_counts else 0
            },
            "timestamps": {
                "first": filtered_records[0]["timestamp"] if filtered_records else None,
                "last": filtered_records[-1]["timestamp"] if filtered_records else None
            }
        }

        return {
            "status": "success",
            "statistics": statistics
        }
    except Exception as e:
        logger.error(f"获取统计数据时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器处理错误: {str(e)}")


@app.post("/api/density/predict")
async def predict_density(request: PredictionRequest = Body(...)):
    """预测未来人流密度"""
    try:
        global predictor
        if predictor is None or predictor.model is None:
            raise HTTPException(status_code=503, detail="预测模型未加载或不可用")

        # 执行预测
        prediction_result = predictor.predict_next_hours(
            location=request.location,
            hours_to_predict=request.hours
        )

        if prediction_result is None:
            raise HTTPException(status_code=400, detail="预测失败，可能是历史数据不足")

        return {
            "status": "success",
            "prediction": prediction_result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测人流密度时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测处理错误: {str(e)}")

# 启动服务器函数


def start_server(host="0.0.0.0", port=5000):
    """启动FastAPI服务器"""
    uvicorn.run(app, host=host, port=port)


# 如果直接运行此文件
if __name__ == "__main__":
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="人流密度监测服务器")
    parser.add_argument("--host", type=str,
                        default="127.0.0.1", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=5000, help="服务器端口")

    args = parser.parse_args()

    print(f"启动服务器 - 监听 {args.host}:{args.port}")
    start_server(args.host, args.port)
