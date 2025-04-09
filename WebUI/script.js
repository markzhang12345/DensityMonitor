// 配置
const API_BASE_URL = "http://localhost:5000/api";
const REFRESH_INTERVAL = 30000; // 30秒刷新一次数据

// 全局变量
let locations = [];
let selectedLocation = "";
let heatmapInstance;
let predictionChart = null;
let refreshInterval;

// 初始化函数
document.addEventListener("DOMContentLoaded", function () {
  initializePage();
  setupEventListeners();
});

// 页面初始化
function initializePage() {
  // 显示当前时间
  updateCurrentTime();
  setInterval(updateCurrentTime, 1000);

  // 初始化热图
  initializeHeatmap();

  // 加载初始数据
  loadAllData();

  // 设置自动刷新
  refreshInterval = setInterval(loadAllData, REFRESH_INTERVAL);
}

// 设置事件监听器
function setupEventListeners() {
  // 刷新按钮
  document.getElementById("refreshBtn").addEventListener("click", loadAllData);

  // 时间范围选择
  document.getElementById("timeRange").addEventListener("change", function () {
    loadStatistics();
  });

  // 位置选择器
  document.getElementById("locationSelector").addEventListener("change", function () {
    selectedLocation = this.value;
    loadStatistics();
    updateHeatmap();
  });

  // 预测按钮
  document.getElementById("predictionBtn").addEventListener("click", function () {
    openPredictionModal();
  });

  // 关闭模态窗口按钮
  document.querySelector(".close-modal").addEventListener("click", function () {
    document.getElementById("predictionModal").style.display = "none";
  });

  // 点击模态窗口外部关闭
  window.addEventListener("click", function (event) {
    if (event.target === document.getElementById("predictionModal")) {
      document.getElementById("predictionModal").style.display = "none";
    }
  });

  // 运行预测按钮
  document.getElementById("runPredictionBtn").addEventListener("click", function () {
    runPrediction();
  });
}

// 更新当前时间显示
function updateCurrentTime() {
  const now = new Date();
  document.getElementById("currentTime").textContent = now.toLocaleString();
}

// 加载所有数据
function loadAllData() {
  loadRecentData()
    .then(() => {
      return loadStatistics();
    })
    .then(() => {
      updateHeatmap();
    })
    .catch((error) => {
      console.error("加载数据时发生错误:", error);
      showToast("加载数据失败，请检查网络连接", "error");
    });
}

// 加载最近数据
async function loadRecentData() {
  try {
    const response = await fetch(`${API_BASE_URL}/density/recent?limit=50`);
    if (!response.ok) {
      throw new Error("获取最近数据失败");
    }

    const data = await response.json();
    if (data.data && data.data.length > 0) {
      processLocationData(data.data);
      return data;
    }
    return null;
  } catch (error) {
    console.error("获取最近数据时出错:", error);
    return null;
  }
}

// 处理位置数据
function processLocationData(records) {
  if (!records || records.length === 0) return;

  // 提取唯一位置
  const uniqueLocations = [...new Set(records.map((record) => record.location))];

  // 按位置分组数据
  const locationData = uniqueLocations.map((location) => {
    const locationRecords = records.filter((record) => record.location === location);
    const lastRecord = locationRecords[locationRecords.length - 1];

    return {
      name: location,
      latestDensity: lastRecord.filtered_density,
      latestPeople: lastRecord.estimated_people,
      records: locationRecords,
      lastUpdated: new Date(lastRecord.timestamp),
    };
  });

  // 保存位置数据
  locations = locationData;

  // 更新位置列表UI
  updateLocationList();

  // 更新位置选择器
  updateLocationSelectors();
}

// 更新位置列表
function updateLocationList() {
  const locationList = document.getElementById("locationList");

  // 清空现有内容
  locationList.innerHTML = "";

  if (locations.length === 0) {
    locationList.innerHTML = '<div class="no-data">暂无位置数据</div>';
    return;
  }

  // 按密度降序排序
  locations.sort((a, b) => b.latestDensity - a.latestDensity);

  // 添加位置项
  locations.forEach((location) => {
    const locationItem = document.createElement("div");
    locationItem.className = "location-item";
    if (selectedLocation === location.name) {
      locationItem.classList.add("active");
    }

    // 确定密度等级
    let densityClass = "density-low";
    if (location.latestDensity >= 1.0) {
      densityClass = "density-high";
    } else if (location.latestDensity >= 0.5) {
      densityClass = "density-medium";
    }

    locationItem.innerHTML = `
            <div class="location-name">${location.name}</div>
            <div class="location-density">
                <span>${location.latestDensity.toFixed(2)} 人/㎡</span>
                <div class="density-indicator ${densityClass}"></div>
            </div>
        `;

    // 点击事件：选择该位置
    locationItem.addEventListener("click", function () {
      document.querySelectorAll(".location-item").forEach((item) => item.classList.remove("active"));
      this.classList.add("active");
      selectedLocation = location.name;

      // 更新选择器
      document.getElementById("locationSelector").value = selectedLocation;

      // 重新加载统计数据和热图
      loadStatistics();
      updateHeatmap();
    });

    locationList.appendChild(locationItem);
  });
}

// 更新位置选择器
function updateLocationSelectors() {
  const selectors = [document.getElementById("locationSelector"), document.getElementById("predictionLocation")];

  selectors.forEach((selector) => {
    // 保存当前选择的值
    const currentValue = selector.value;

    // 清空现有选项（保留"所有位置"选项）
    while (selector.options.length > 1) {
      selector.remove(1);
    }

    // 添加位置选项
    locations.forEach((location) => {
      const option = document.createElement("option");
      option.value = location.name;
      option.textContent = location.name;
      selector.appendChild(option);
    });

    // 恢复之前的选择（如果有效）
    if (currentValue && locations.some((loc) => loc.name === currentValue)) {
      selector.value = currentValue;
    }
  });
}

// 加载统计数据
async function loadStatistics() {
  try {
    const timeRange = document.getElementById("timeRange").value;
    let url = `${API_BASE_URL}/density/statistics?hours=${timeRange}`;

    if (selectedLocation) {
      url += `&location=${encodeURIComponent(selectedLocation)}`;
    }

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error("获取统计数据失败");
    }

    const data = await response.json();
    if (data.statistics) {
      updateStatisticsUI(data.statistics);
      return data.statistics;
    }
    return null;
  } catch (error) {
    console.error("获取统计数据时出错:", error);
    return null;
  }
}

// 更新统计UI
function updateStatisticsUI(statistics) {
  if (!statistics) return;

  // 更新密度统计
  document.getElementById("currentDensity").textContent = statistics.density.current.toFixed(2);
  document.getElementById("avgDensity").textContent = statistics.density.average.toFixed(2);
  document.getElementById("maxDensity").textContent = statistics.density.max.toFixed(2);

  // 更新人数统计
  document.getElementById("currentPeople").textContent = Math.round(statistics.people.current);

  // 根据密度级别更改颜色
  const currentDensityElement = document.getElementById("currentDensity");
  if (statistics.density.current >= 1.0) {
    currentDensityElement.style.color = "#dc3545"; // 红色
  } else if (statistics.density.current >= 0.5) {
    currentDensityElement.style.color = "#ffc107"; // 黄色
  } else {
    currentDensityElement.style.color = "#28a745"; // 绿色
  }
}

// 初始化热图
function initializeHeatmap() {
  const heatmapContainer = document.getElementById("heatmapContainer");

  // 创建热图实例
  heatmapInstance = h337.create({
    container: heatmapContainer,
    radius: 40,
    maxOpacity: 0.8,
    minOpacity: 0.1,
    blur: 0.85,
    gradient: {
      0.0: "blue",
      0.3: "green",
      0.6: "yellow",
      1.0: "red",
    },
  });
}

// 更新热图数据
function updateHeatmap() {
  if (!locations || locations.length === 0) return;

  // 如果选择了特定位置，则只显示该位置
  let filteredLocations = locations;
  if (selectedLocation) {
    filteredLocations = locations.filter((loc) => loc.name === selectedLocation);
  }

  // 容器尺寸
  const container = document.getElementById("heatmapContainer");
  const width = container.clientWidth;
  const height = container.clientHeight;

  // 生成热图数据点
  const points = [];
  const maxValue = Math.max(...filteredLocations.map((loc) => loc.latestDensity));

  // 为每个位置生成数据点
  filteredLocations.forEach((location, index) => {
    // 计算位置坐标（这里使用简单分布，实际应用可能需要真实坐标）
    const x = Math.round(((index + 1) * width) / (filteredLocations.length + 1));
    const y = Math.round(height / 2 + (Math.random() - 0.5) * height * 0.6);

    // 添加主要数据点
    points.push({
      x: x,
      y: y,
      value: location.latestDensity,
      radius: 30 + location.latestDensity * 20,
    });

    // 在周围添加一些随机点，形成更自然的热区
    const numRandomPoints = Math.ceil(location.latestDensity * 10);
    for (let i = 0; i < numRandomPoints; i++) {
      const randomX = x + (Math.random() - 0.5) * 100;
      const randomY = y + (Math.random() - 0.5) * 100;
      const randomValue = location.latestDensity * (0.3 + Math.random() * 0.7);

      points.push({
        x: randomX,
        y: randomY,
        value: randomValue,
        radius: 20 + randomValue * 10,
      });
    }

    // 添加位置标签
    const label = document.createElement("div");
    label.className = "location-label";
    label.style.position = "absolute";
    label.style.left = `${x - 50}px`;
    label.style.top = `${y - 50}px`;
    label.style.width = "100px";
    label.style.textAlign = "center";
    label.style.color = "#fff";
    label.style.textShadow = "1px 1px 3px rgba(0,0,0,0.8)";
    label.style.fontSize = "12px";
    label.style.fontWeight = "bold";
    label.style.pointerEvents = "none";
    label.innerHTML = `${location.name}<br>${location.latestDensity.toFixed(2)} 人/㎡`;

    // 清除旧标签
    const oldLabels = container.querySelectorAll(".location-label");
    oldLabels.forEach((label) => label.remove());

    container.appendChild(label);
  });

  // 设置热图数据
  heatmapInstance.setData({
    max: Math.max(1.5, maxValue),
    data: points,
  });
}

// 打开预测模态窗口
function openPredictionModal() {
  document.getElementById("predictionModal").style.display = "block";

  // 如果已经选择了位置，同步到预测位置选择器
  if (selectedLocation) {
    document.getElementById("predictionLocation").value = selectedLocation;
  }

  // 运行初始预测
  runPrediction();
}

// 运行预测
async function runPrediction() {
  try {
    const location = document.getElementById("predictionLocation").value;
    const hours = document.getElementById("predictionHours").value;

    // 显示加载状态
    const chartContainer = document.querySelector(".prediction-chart-container");
    chartContainer.innerHTML = `
            <div class="loading-spinner">
                <i class="fas fa-circle-notch fa-spin"></i>
                <span>正在生成预测...</span>
            </div>
        `;

    // 准备请求数据
    const requestData = {
      location: location,
      hours: parseInt(hours),
    };

    // 发送预测请求
    const response = await fetch(`${API_BASE_URL}/density/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestData),
    });

    if (!response.ok) {
      throw new Error("预测请求失败");
    }

    const result = await response.json();

    // 恢复图表容器
    chartContainer.innerHTML = '<canvas id="predictionChart"></canvas>';

    // 显示预测结果
    displayPredictionChart(result.prediction);
  } catch (error) {
    console.error("预测失败:", error);
    showToast("生成预测失败，请稍后重试", "error");

    // 恢复图表容器
    const chartContainer = document.querySelector(".prediction-chart-container");
    chartContainer.innerHTML = `
            <div class="prediction-error">
                <p><i class="fas fa-exclamation-triangle"></i> 预测失败: ${error.message}</p>
                <p>请确保有足够的历史数据用于预测</p>
            </div>
        `;
  }
}

// 显示预测图表
function displayPredictionChart(prediction) {
  if (!prediction || !prediction.predictions || prediction.predictions.length === 0) {
    return;
  }

  // 准备图表数据
  const labels = prediction.predictions.map((p) => {
    const date = new Date(p.timestamp);
    return `${date.getHours()}:00`;
  });

  const densityData = prediction.predictions.map((p) => p.predicted_density);

  // 计算时段（早上、下午、晚上）
  const timeOfDayColors = prediction.predictions.map((p) => {
    const hour = new Date(p.timestamp).getHours();
    if (hour >= 6 && hour < 12) return "rgba(255, 193, 7, 0.8)"; // 早上 - 黄色
    if (hour >= 12 && hour < 18) return "rgba(0, 123, 255, 0.8)"; // 下午 - 蓝色
    return "rgba(108, 117, 125, 0.8)"; // 晚上 - 灰色
  });

  // 销毁旧图表
  if (predictionChart) {
    predictionChart.destroy();
  }

  // 创建新图表
  const ctx = document.getElementById("predictionChart").getContext("2d");
  predictionChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          type: "line",
          label: "预测密度趋势",
          data: densityData,
          fill: false,
          borderColor: "rgba(75, 192, 192, 1)",
          tension: 0.4,
          pointRadius: 3,
          pointHoverRadius: 6,
          pointBackgroundColor: "rgba(75, 192, 192, 1)",
          order: 0,
        },
        {
          type: "bar",
          label: "预测密度值",
          data: densityData,
          backgroundColor: timeOfDayColors,
          order: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: `${prediction.location} - 人流密度预测`,
          font: {
            size: 16,
          },
        },
        tooltip: {
          mode: "index",
          intersect: false,
          callbacks: {
            label: function (context) {
              return `预测密度: ${context.raw.toFixed(2)} 人/㎡`;
            },
          },
        },
        legend: {
          position: "top",
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "时间",
          },
        },
        y: {
          title: {
            display: true,
            text: "人流密度 (人/平方米)",
          },
          beginAtZero: true,
          suggestedMin: 0,
          suggestedMax: Math.max(...densityData) * 1.2,
        },
      },
    },
  });
}

// 显示提示消息
function showToast(message, type = "info") {
  // 检查是否已存在Toast容器
  let toastContainer = document.querySelector(".toast-container");
  if (!toastContainer) {
    toastContainer = document.createElement("div");
    toastContainer.className = "toast-container";
    toastContainer.style.position = "fixed";
    toastContainer.style.top = "20px";
    toastContainer.style.right = "20px";
    toastContainer.style.zIndex = "9999";
    document.body.appendChild(toastContainer);
  }

  // 创建新的Toast
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.style.backgroundColor = type === "error" ? "#f8d7da" : "#d1e7dd";
  toast.style.color = type === "error" ? "#842029" : "#0f5132";
  toast.style.padding = "1rem";
  toast.style.marginBottom = "10px";
  toast.style.borderRadius = "4px";
  toast.style.boxShadow = "0 2px 5px rgba(0,0,0,0.2)";
  toast.style.minWidth = "250px";

  // 添加图标
  const icon = document.createElement("i");
  icon.className = type === "error" ? "fas fa-exclamation-circle" : "fas fa-info-circle";
  icon.style.marginRight = "10px";
  toast.appendChild(icon);

  // 添加消息文本
  const messageText = document.createTextNode(message);
  toast.appendChild(messageText);

  // 添加到容器
  toastContainer.appendChild(toast);

  // 自动消失
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.5s ease";
    setTimeout(() => {
      toastContainer.removeChild(toast);
      if (toastContainer.children.length === 0) {
        document.body.removeChild(toastContainer);
      }
    }, 500);
  }, 3000);
}

// 实时位置模拟（仅用于演示）
function simulateRealTimeData() {
  if (!locations || locations.length === 0) return;

  // 对每个位置生成随机变化
  locations.forEach((location) => {
    // 随机变化率 (-5% 到 +5%)
    const changeRate = 1 + (Math.random() - 0.5) * 0.1;

    // 应用变化
    location.latestDensity *= changeRate;

    // 确保值在合理范围内
    location.latestDensity = Math.max(0.05, Math.min(2.0, location.latestDensity));

    // 更新人数
    const area = location.records[0].area_size;
    location.latestPeople = location.latestDensity * area;
  });

  // 更新UI
  updateLocationList();
  updateHeatmap();

  // 如果选择了特定位置，更新统计数据
  if (selectedLocation) {
    const location = locations.find((loc) => loc.name === selectedLocation);
    if (location) {
      const simulatedStats = {
        density: {
          current: location.latestDensity,
          average: location.latestDensity * 0.9,
          max: location.latestDensity * 1.2,
          min: location.latestDensity * 0.7,
        },
        people: {
          current: location.latestPeople,
          average: location.latestPeople * 0.9,
          max: location.latestPeople * 1.2,
          min: location.latestPeople * 0.7,
        },
      };
      updateStatisticsUI(simulatedStats);
    }
  }
}

// 演示模式 - 如果没有实际数据，启用此功能
function enableDemoMode() {
  // 创建虚拟位置数据
  if (!locations || locations.length === 0) {
    const demoLocations = [
      { name: "图书馆入口", latestDensity: 0.4, latestPeople: 80 },
      { name: "学生中心", latestDensity: 0.7, latestPeople: 140 },
      { name: "食堂一楼", latestDensity: 1.2, latestPeople: 240 },
      { name: "教学楼大厅", latestDensity: 0.3, latestPeople: 60 },
      { name: "体育馆", latestDensity: 0.2, latestPeople: 40 },
    ];

    // 为演示位置添加必要的属性
    locations = demoLocations.map((location) => {
      return {
        ...location,
        lastUpdated: new Date(),
        records: [
          {
            area_size: location.latestPeople / location.latestDensity,
          },
        ],
      };
    });

    // 更新UI
    updateLocationList();
    updateLocationSelectors();
    updateHeatmap();
  }

  // 启动数据模拟
  setInterval(simulateRealTimeData, 5000);

  // 显示演示模式通知
  showToast("演示模式已启动: 数据为模拟生成", "info");
}

// 检测是否需要启用演示模式
setTimeout(() => {
  if (!locations || locations.length === 0) {
    enableDemoMode();
  }
}, 5000);
