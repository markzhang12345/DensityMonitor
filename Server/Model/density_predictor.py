import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("density_prediction.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DensityPredictor")

# 模型文件和数据路径
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("./density_data")


class DensityPredictor:
    """人流密度预测模型类"""

    def __init__(self, model_name="density_lstm_model", sequence_length=24):
        """
        初始化预测器

        参数:
        - model_name: 模型名称
        - sequence_length: 用于预测的历史序列长度（小时）
        """
        self.model_name = model_name
        self.model_path = MODEL_DIR / f"{model_name}.h5"
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # 尝试加载已有模型
        if self.model_path.exists():
            try:
                self.model = load_model(self.model_path)
                logger.info(f"成功加载模型: {self.model_path}")
            except Exception as e:
                logger.error(f"加载模型失败: {str(e)}")
                self.model = None

    def _load_data(self, location=None, days=30):
        """
        加载历史数据

        参数:
        - location: 特定位置，如果为None则加载所有位置
        - days: 加载过去多少天的数据

        返回:
        - DataFrame包含时间戳和密度数据
        """
        today = datetime.datetime.now()
        all_data = []

        # 计算起始日期
        start_date = today - datetime.timedelta(days=days)

        # 获取数据文件列表
        data_files = list(DATA_DIR.glob("density_data_*.json"))
        data_files.sort()  # 按日期排序

        for file_path in data_files:
            try:
                # 从文件名解析日期
                date_str = file_path.stem.split('_')[-1]
                file_date = datetime.datetime.strptime(date_str, "%Y%m%d")

                # 如果文件日期在范围内，加载数据
                if file_date >= start_date:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)

                        # 过滤位置（如果指定）
                        if location:
                            file_data = [
                                r for r in file_data if r['location'] == location]

                        all_data.extend(file_data)
            except Exception as e:
                logger.warning(f"读取文件 {file_path} 时出错: {str(e)}")

        if not all_data:
            logger.warning(f"没有找到符合条件的数据")
            return None

        # 转换为DataFrame
        df = pd.DataFrame(all_data)

        # 确保时间戳格式正确并排序
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # 重采样到小时级别（取平均值）
        df.set_index('timestamp', inplace=True)
        hourly_df = df['filtered_density'].resample('H').mean()

        # 处理缺失值
        hourly_df = hourly_df.interpolate(method='linear')

        return hourly_df.reset_index()

    def _prepare_sequences(self, data):
        """
        准备训练序列

        参数:
        - data: 时序数据

        返回:
        - X: 输入序列
        - y: 目标值
        """
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))

        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length, 0])
            y.append(scaled_data[i + self.sequence_length, 0])

        return np.array(X).reshape(-1, self.sequence_length, 1), np.array(y)

    def train(self, location=None, epochs=50, batch_size=32, validation_split=0.2):
        """
        训练LSTM模型

        参数:
        - location: 特定位置，如果为None则使用所有位置的数据
        - epochs: 训练轮数
        - batch_size: 批量大小
        - validation_split: 验证集比例

        返回:
        - 训练历史
        """
        # 加载数据
        df = self._load_data(location=location)
        if df is None or len(df) < self.sequence_length + 10:
            logger.error("数据不足，无法训练模型")
            return None

        logger.info(f"加载了 {len(df)} 条数据记录用于训练")

        # 准备序列
        X, y = self._prepare_sequences(df['filtered_density'])

        # 创建轻量级LSTM模型
        self.model = Sequential()
        self.model.add(LSTM(32, input_shape=(
            self.sequence_length, 1), return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(16))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))

        # 编译模型
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # 设置回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10,
                          restore_best_weights=True),
            ModelCheckpoint(filepath=self.model_path, save_best_only=True)
        ]

        # 训练模型
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        logger.info(f"模型训练完成，已保存到 {self.model_path}")
        return history

    def predict_next_hours(self, location=None, hours_to_predict=24):
        """
        预测未来几小时的人流密度

        参数:
        - location: 特定位置，如果为None则使用所有位置的数据
        - hours_to_predict: 预测未来多少小时

        返回:
        - 预测结果字典，包含时间戳和预测密度
        """
        if self.model is None:
            logger.error("模型未加载，无法进行预测")
            return None

        # 加载最近数据
        df = self._load_data(location=location, days=3)  # 只需要最近几天的数据
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

    def evaluate(self, location=None, test_days=7):
        """
        评估模型性能

        参数:
        - location: 特定位置，如果为None则使用所有位置的数据
        - test_days: 用于测试的天数

        返回:
        - 评估指标字典
        """
        if self.model is None:
            logger.error("模型未加载，无法评估")
            return None

        # 加载测试数据
        df = self._load_data(location=location, days=test_days)
        if df is None or len(df) < self.sequence_length + 10:
            logger.error("测试数据不足")
            return None

        # 准备测试序列
        X_test, y_test = self._prepare_sequences(df['filtered_density'])

        # 预测
        y_pred = self.model.predict(X_test)

        # 转换回原始比例
        y_test_orig = self.scaler.inverse_transform(
            y_test.reshape(-1, 1)).flatten()
        y_pred_orig = self.scaler.inverse_transform(y_pred).flatten()

        # 计算评估指标
        mse = np.mean((y_test_orig - y_pred_orig) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_orig - y_pred_orig))

        # 计算均值和标准差
        mean_actual = np.mean(y_test_orig)
        std_actual = np.std(y_test_orig)

        # 计算MAPE (Mean Absolute Percentage Error)
        # 避免除以零
        mape_array = np.abs((y_test_orig - y_pred_orig) /
                            (y_test_orig + 1e-10)) * 100
        mape = np.mean(mape_array)

        return {
            "location": location if location else "所有位置",
            "test_samples": len(y_test_orig),
            "metrics": {
                "MSE": float(mse),
                "RMSE": float(rmse),
                "MAE": float(mae),
                "MAPE": float(mape),
                "mean_actual": float(mean_actual),
                "std_actual": float(std_actual)
            }
        }

    def plot_predictions(self, location=None, past_hours=24, future_hours=24, save_path=None):
        """
        绘制预测结果图表

        参数:
        - location: 特定位置
        - past_hours: 显示过去多少小时的实际数据
        - future_hours: 预测未来多少小时
        - save_path: 保存图片的路径，如果为None则显示图片

        返回:
        - 无
        """
        # 获取历史数据
        df = self._load_data(location=location, days=3)
        if df is None or len(df) < past_hours:
            logger.error("历史数据不足，无法绘图")
            return

        # 获取预测
        predictions = self.predict_next_hours(
            location=location, hours_to_predict=future_hours)
        if predictions is None:
            logger.error("预测失败，无法绘图")
            return

        # 准备数据
        historical_data = df.iloc[-past_hours:]

        # 创建图表
        plt.figure(figsize=(12, 6))

        # 绘制历史数据
        plt.plot(historical_data['timestamp'],
                 historical_data['filtered_density'],
                 'b-', label='历史密度')

        # 绘制预测数据
        pred_times = [datetime.datetime.strptime(p['timestamp'], "%Y-%m-%dT%H:%M:%S")
                      for p in predictions['predictions']]
        pred_values = [p['predicted_density']
                       for p in predictions['predictions']]
        plt.plot(pred_times, pred_values, 'r--', label='预测密度')

        # 添加当前时间的垂直线
        current_time = datetime.datetime.now()
        plt.axvline(x=current_time, color='g', linestyle='-', label='当前时间')

        # 设置图表属性
        location_name = location if location else "所有位置"
        plt.title(f'{location_name}人流密度预测')
        plt.xlabel('时间')
        plt.ylabel('人流密度 (人/平方米)')
        plt.legend()
        plt.grid(True)

        # 格式化时间轴
        plt.gcf().autofmt_xdate()

        # 保存或显示
        if save_path:
            plt.savefig(save_path)
            logger.info(f"图表已保存到 {save_path}")
        else:
            plt.show()

        plt.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="人流密度预测工具")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--predict", action="store_true", help="进行预测")
    parser.add_argument("--evaluate", action="store_true", help="评估模型")
    parser.add_argument("--plot", action="store_true", help="绘制预测图表")
    parser.add_argument("--location", type=str, default=None, help="指定位置")
    parser.add_argument("--hours", type=int, default=24, help="预测未来小时数")
    parser.add_argument("--save", type=str, default=None, help="保存图表路径")

    args = parser.parse_args()

    # 创建预测器实例
    predictor = DensityPredictor()

    if args.train:
        logger.info("开始训练模型...")
        history = predictor.train(location=args.location)
        if history:
            logger.info("模型训练完成")

    if args.predict:
        logger.info(f"预测未来 {args.hours} 小时的人流密度...")
        predictions = predictor.predict_next_hours(
            location=args.location,
            hours_to_predict=args.hours
        )
        if predictions:
            print(json.dumps(predictions, indent=2, ensure_ascii=False))

    if args.evaluate:
        logger.info("评估模型性能...")
        metrics = predictor.evaluate(location=args.location)
        if metrics:
            print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.plot:
        logger.info("绘制预测图表...")
        predictor.plot_predictions(
            location=args.location,
            future_hours=args.hours,
            save_path=args.save
        )


if __name__ == "__main__":
    main()
