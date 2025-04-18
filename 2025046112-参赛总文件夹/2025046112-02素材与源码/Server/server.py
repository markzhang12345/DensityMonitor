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
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DensityMonitor")


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

density_records = []


class DensityPredictor:
    """人流密度预测模型类 - SARIMA版本"""

    def __init__(self, models_dir="./Model/models"):
        """
        初始化SARIMA预测器

        参数:
        - models_dir: 包含预训练SARIMA模型的目录
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.sequence_length = 24
        self.model = None

        # 加载所有预训练的SARIMA模型
        try:
            import pickle
            try:
                import joblib
            except ImportError:
                joblib = None
                logger.warning("未找到joblib库，将仅使用pickle加载模型")

            if not self.models_dir.exists():
                logger.error(f"模型目录不存在: {self.models_dir}")
                return

            model_files = list(self.models_dir.glob(
                "density_sarima_model_*.pkl"))
            if not model_files:
                logger.error(f"未找到SARIMA模型文件，路径: {self.models_dir}")
                return

            for model_path in model_files:
                location = model_path.stem.replace("density_sarima_model_", "")
                try:
                    model = None
                    if joblib:
                        try:
                            model = joblib.load(model_path)
                        except Exception as e:
                            logger.warning(
                                f"使用joblib加载模型失败，尝试pickle: {str(e)}")

                    if model is None:
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)

                    self.models[location] = model
                    logger.info(f"成功加载SARIMA模型: {location}")
                except Exception as e:
                    logger.error(f"加载SARIMA模型失败 {location}: {str(e)}")

            if not self.models:
                logger.error("没有成功加载任何SARIMA模型")
            else:
                self.model = True
                logger.info(f"成功加载了 {len(self.models)} 个SARIMA模型")
        except Exception as e:
            logger.error(f"初始化SARIMA预测器时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def _load_data(self, location=None, days=7):
        """加载历史数据"""
        today = datetime.datetime.now()
        all_data = []

        memory_data = [record for record in density_records if
                       (location is None or record["location"] == location)]

        data_files = sorted(DATA_DIR.glob("density_data_*.json"), reverse=True)
        for file_path in data_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)

                    if location:
                        file_data = [
                            r for r in file_data if r['location'] == location]

                    all_data.extend(file_data)
            except Exception as e:
                logger.warning(f"读取文件 {file_path} 时出错: {str(e)}")

        all_data.extend(memory_data)

        if not all_data:
            logger.warning(f"没有找到符合条件的数据")
            return None

        df = pd.DataFrame(all_data)

        if df.empty:
            logger.warning("数据转换为DataFrame后为空")
            return None

        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            df.set_index('timestamp', inplace=True)
            hourly_df = df['filtered_density'].resample('h').mean()

            hourly_df = hourly_df.interpolate(method='linear')

            return hourly_df
        except Exception as e:
            logger.error(f"处理时间序列数据时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _map_location_to_model(self, location):
        """将用户输入的位置映射到对应的模型名称"""
        if not location:
            return "center"

        location_lower = location.lower()

        if "图书馆入口" in location_lower:
            return "library"
        elif "电影院" in location_lower or "影院" in location_lower:
            return "cinema"
        elif "餐饮区" in location_lower or "食堂" in location_lower or "美食" in location_lower:
            return "food"
        elif "商场入口" in location_lower or "商场" in location_lower:
            return "market"
        elif "地铁站" in location_lower or "站台" in location_lower:
            return "subway"
        else:
            return "center"

    def predict_next_hours(self, location=None, hours_to_predict=24):
        """使用SARIMA模型预测未来几小时的人流密度"""
        try:
            if not self.models:
                logger.error("SARIMA模型未加载，无法进行预测")
                return None

            import pandas as pd
            import numpy as np

            try:
                import statsmodels.api as sm
            except ImportError:
                logger.error("未找到statsmodels库，无法进行SARIMA预测")
                return None

            model_key = self._map_location_to_model(location)
            if model_key not in self.models:
                logger.warning(
                    f"未找到位置 '{location}' 对应的模型 '{model_key}'，尝试使用其他模型")
                if self.models:
                    model_key = next(iter(self.models.keys()))
                else:
                    logger.error("没有可用的SARIMA模型")
                    return None

            logger.info(f"使用模型 '{model_key}' 预测位置 '{location}'")

            df = self._load_data(location=location, days=7)
            if df is None or len(df) < 24:  # 需要至少一天的数据
                logger.error("数据不足，无法进行预测")
                return None

            model = self.models[model_key]

            # 使用SARIMA模型预测
            try:
                forecast = model.get_forecast(steps=hours_to_predict)
                predicted_mean = forecast.predicted_mean

                if isinstance(predicted_mean, pd.Series):
                    predictions = predicted_mean.values
                else:
                    predictions = predicted_mean

                predictions = np.maximum(predictions, 0)

                # 生成时间戳
                last_timestamp = df.index[-1]
                timestamps = [
                    last_timestamp + datetime.timedelta(hours=i+1) for i in range(hours_to_predict)]

                result = {
                    "location": location if location else "所有位置",
                    "model_used": model_key,
                    "predictions": [
                        {
                            "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                            "predicted_density": float(density),
                            "hour": timestamp.hour
                        }
                        for timestamp, density in zip(timestamps, predictions)
                    ]
                }

                return result

            except Exception as e:
                logger.error(f"SARIMA预测过程出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return None

        except Exception as e:
            logger.error(f"预测过程发生未知错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None


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

        combined_data = existing_data + data_list

        temp_file = file_path.with_suffix('.tmp')
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)

        temp_file.replace(file_path)
        logger.info(f"成功保存 {len(data_list)} 条数据到 {file_path}")
    except Exception as e:
        logger.error(f"保存数据时出错: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("应用服务启动...")
    global predictor
    try:
        predictor = DensityPredictor(models_dir="./Model/models")
        if not predictor.models:
            logger.warning("SARIMA预测模型未能正确加载，预测功能将不可用")
        else:
            logger.info(f"成功加载 {len(predictor.models)} 个SARIMA预测模型")
    except Exception as e:
        logger.error(f"初始化SARIMA预测器时出错: {str(e)}")
        predictor = None

    yield

    logger.info("应用关闭中...")
    if density_records:
        try:
            save_data_to_file(density_records)
            logger.info(f"应用关闭，已保存 {len(density_records)} 条数据")
        except Exception as e:
            logger.error(f"应用关闭时保存数据失败: {str(e)}")


app = FastAPI(
    title="人流密度监测服务",
    description="接收、存储和查询人流密度数据的API服务",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        record = data.model_dump()
        record["received_at"] = datetime.datetime.now().isoformat()

        density_records.append(record)

        if len(density_records) >= 100:
            records_to_save = density_records.copy()
            density_records.clear()
            try:
                save_data_to_file(records_to_save)
            except Exception as e:
                density_records.extend(records_to_save)
                logger.error(f"保存数据失败: {str(e)}")

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
        recent_records = density_records[-limit:] if density_records else []

        if len(recent_records) < limit:
            needed = limit - len(recent_records)
            try:
                data_files = sorted(DATA_DIR.glob(
                    "density_data_*.json"), reverse=True)
                if data_files:
                    with open(data_files[0], "r", encoding="utf-8") as f:
                        file_records = json.load(f)
                        file_records = file_records[-needed:] if len(
                            file_records) > needed else file_records
                        recent_records = file_records + recent_records
            except Exception as e:
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
        now = datetime.datetime.now()
        start_time = (now - datetime.timedelta(hours=hours)).isoformat()

        all_records = density_records.copy()

        try:
            data_files = sorted(DATA_DIR.glob(
                "density_data_*.json"), reverse=True)
            for file_path in data_files:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_records = json.load(f)
                    all_records.extend(file_records)
        except Exception as e:
            logger.error(f"读取统计数据时出错: {str(e)}")

        filtered_records = []
        for record in all_records:
            if record["timestamp"] >= start_time:
                if location is None or record["location"] == location:
                    filtered_records.append(record)

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
        if predictor is None or not predictor.models:
            raise HTTPException(status_code=503, detail="预测模型未加载或不可用")

        # 执行预测
        prediction_result = predictor.predict_next_hours(
            location=request.location,
            hours_to_predict=request.hours
        )

        if prediction_result is None:
            raise HTTPException(
                status_code=400, detail="预测失败，可能是历史数据不足或无法找到合适的模型")

        return {
            "status": "success",
            "prediction": prediction_result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测人流密度时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"预测处理错误: {str(e)}")


def start_server(host="0.0.0.0", port=5000):
    """启动FastAPI服务器"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="人流密度监测服务器")
    parser.add_argument("--host", type=str,
                        default="127.0.0.1", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=5000, help="服务器端口")

    args = parser.parse_args()

    print(f"启动服务器 - 监听 {args.host}:{args.port}")
    start_server(args.host, args.port)
