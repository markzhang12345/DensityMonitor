import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import datetime
import logging
import pickle

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

    def __init__(self, model_name="density_sarima_model", order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)):
        """
        初始化预测器

        参数:
        - model_name: 模型名称
        - order: SARIMA模型的非季节性参数 (p,d,q)
        - seasonal_order: SARIMA模型的季节性参数 (P,D,Q,s)
        """
        self.model_name = model_name
        self.model_path = MODEL_DIR / f"{model_name}.pkl"
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # 尝试加载已有模型
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
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

        return hourly_df

    def train(self, location=None):
        """
        训练SARIMA模型

        参数:
        - location: 特定位置，如果为None则使用所有位置的数据

        返回:
        - 训练结果
        """
        # 加载数据
        series = self._load_data(location=location)
        if series is None or len(series) < 48:  # 需要足够的数据
            logger.error("数据不足，无法训练模型")
            return None

        logger.info(f"加载了 {len(series)} 条数据记录用于训练")

        # 对数据进行缩放
        scaled_data = self.scaler.fit_transform(
            series.values.reshape(-1, 1)).flatten()
        scaled_series = pd.Series(scaled_data, index=series.index)

        # 创建SARIMA模型
        try:
            self.model = SARIMAX(
                scaled_series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            # 拟合模型
            results = self.model.fit(disp=False)

            # 保存模型
            with open(self.model_path, 'wb') as f:
                pickle.dump(results, f)

            logger.info(f"模型训练完成，已保存到 {self.model_path}")
            return results
        except Exception as e:
            logger.error(f"训练模型时发生错误: {str(e)}")
            return None

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

        # 加载历史数据
        series = self._load_data(location=location, days=7)  # 使用最近7天的数据
        if series is None or len(series) < 24:
            logger.error("数据不足，无法进行预测")
            return None

        # 获取最后一个时间戳
        last_timestamp = series.index[-1]

        # 对历史数据进行缩放
        self.scaler.fit(series.values.reshape(-1, 1))
        scaled_series = pd.Series(
            self.scaler.transform(series.values.reshape(-1, 1)).flatten(),
            index=series.index
        )

        try:
            # 预测未来几小时
            forecast = self.model.predict(
                start=len(scaled_series),
                end=len(scaled_series) + hours_to_predict - 1,
                dynamic=True
            )

            # 转换预测结果回原始比例
            original_predictions = self.scaler.inverse_transform(
                forecast.values.reshape(-1, 1)
            ).flatten()

            # 生成未来时间戳
            timestamps = [
                last_timestamp + datetime.timedelta(hours=i+1) for i in range(hours_to_predict)]

            # 组织返回结果
            result = {
                "location": location if location else "所有位置",
                "predictions": [
                    {
                        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                        "predicted_density": float(max(0, density)),  # 确保密度不为负
                        "hour": timestamp.hour
                    }
                    for timestamp, density in zip(timestamps, original_predictions)
                ]
            }

            return result
        except Exception as e:
            logger.error(f"预测时发生错误: {str(e)}")
            return None

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
        series = self._load_data(
            location=location, days=test_days + 3)  # 多加载一些数据
        if series is None or len(series) < 24 * test_days:
            logger.error("测试数据不足")
            return None

        # 划分训练集和测试集
        train_size = len(series) - 24 * test_days
        train = series.iloc[:train_size]
        test = series.iloc[train_size:]

        # 对数据进行缩放
        self.scaler.fit(train.values.reshape(-1, 1))
        scaled_train = pd.Series(
            self.scaler.transform(train.values.reshape(-1, 1)).flatten(),
            index=train.index
        )

        # 使用训练数据拟合模型
        model = SARIMAX(
            scaled_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)

        # 预测测试集
        forecast = results.get_forecast(steps=len(test))
        forecasted_values = forecast.predicted_mean

        # 转换回原始比例
        y_pred_orig = self.scaler.inverse_transform(
            forecasted_values.values.reshape(-1, 1)
        ).flatten()
        y_test_orig = test.values

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
        series = self._load_data(location=location, days=3)
        if series is None or len(series) < past_hours:
            logger.error("历史数据不足，无法绘图")
            return

        # 获取预测
        predictions = self.predict_next_hours(
            location=location, hours_to_predict=future_hours)
        if predictions is None:
            logger.error("预测失败，无法绘图")
            return

        # 准备数据
        historical_data = series.iloc[-past_hours:].reset_index()

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
        plt.title(f'{location_name}人流密度预测 (SARIMA模型)')
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

    def find_best_parameters(self, location=None, max_p=2, max_d=1, max_q=2, max_P=1, max_D=1, max_Q=1, s=24):
        """
        自动寻找最佳SARIMA参数

        参数:
        - location: 特定位置
        - max_p, max_d, max_q: 非季节性参数的最大值
        - max_P, max_D, max_Q: 季节性参数的最大值
        - s: 季节性周期

        返回:
        - 最佳参数和AIC值
        """
        from itertools import product

        # 加载数据
        series = self._load_data(location=location, days=14)  # 使用两周的数据
        if series is None or len(series) < 72:
            logger.error("数据不足，无法进行参数优化")
            return None

        # 对数据进行缩放
        scaled_data = self.scaler.fit_transform(
            series.values.reshape(-1, 1)).flatten()
        scaled_series = pd.Series(scaled_data, index=series.index)

        # 定义参数网格
        p = range(0, max_p + 1)
        d = range(0, max_d + 1)
        q = range(0, max_q + 1)
        P = range(0, max_P + 1)
        D = range(0, max_D + 1)
        Q = range(0, max_Q + 1)

        # 初始化最佳参数和AIC
        best_aic = float("inf")
        best_params = None

        # 搜索最佳参数
        for param in product(p, d, q, P, D, Q):
            # 构造参数
            order = (param[0], param[1], param[2])
            seasonal_order = (param[3], param[4], param[5], s)

            try:
                # 拟合模型
                model = SARIMAX(
                    scaled_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)

                # 更新最佳参数
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = (order, seasonal_order)

                logger.info(
                    f"SARIMA{order}x{seasonal_order} - AIC: {results.aic}")
            except Exception as e:
                continue

        return {
            "location": location if location else "所有位置",
            "best_params": {
                "order": best_params[0],
                "seasonal_order": best_params[1]
            },
            "aic": float(best_aic)
        }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="人流密度预测工具 (SARIMA模型版)")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--predict", action="store_true", help="进行预测")
    parser.add_argument("--evaluate", action="store_true", help="评估模型")
    parser.add_argument("--plot", action="store_true", help="绘制预测图表")
    parser.add_argument("--find-params", action="store_true", help="寻找最佳参数")
    parser.add_argument("--location", type=str, default=None, help="指定位置")
    parser.add_argument("--hours", type=int, default=24, help="预测未来小时数")
    parser.add_argument("--save", type=str, default=None, help="保存图表路径")
    parser.add_argument("--order", type=str, default="1,1,1",
                        help="SARIMA非季节性参数 (p,d,q)")
    parser.add_argument("--seasonal-order", type=str,
                        default="1,1,1,24", help="SARIMA季节性参数 (P,D,Q,s)")

    args = parser.parse_args()

    # 解析SARIMA参数
    order = tuple(map(int, args.order.split(',')))
    seasonal_order = tuple(map(int, args.seasonal_order.split(',')))

    # 创建预测器实例
    predictor = DensityPredictor(order=order, seasonal_order=seasonal_order)

    if args.find_params:
        logger.info("寻找最佳SARIMA参数...")
        best_params = predictor.find_best_parameters(location=args.location)
        if best_params:
            print(json.dumps(best_params, indent=2, ensure_ascii=False))

            # 更新预测器使用最佳参数
            predictor = DensityPredictor(
                order=best_params["best_params"]["order"],
                seasonal_order=best_params["best_params"]["seasonal_order"]
            )

    if args.train:
        logger.info("开始训练SARIMA模型...")
        results = predictor.train(location=args.location)
        if results:
            logger.info("模型训练完成")
            print(f"模型信息: {results.summary()}")

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
