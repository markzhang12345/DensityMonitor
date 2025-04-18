const API_BASE_URL = "http://localhost:5000/api";
const REFRESH_INTERVAL = 30000; // 30秒刷新一次数据

let locations = [];
let selectedLocation = "";
let heatmapInstance;
let predictionChart = null;
let refreshInterval;

document.addEventListener("DOMContentLoaded", function () {
  initializePage();
  setupEventListeners();
});

function initializePage() {
  updateCurrentTime();
  setInterval(updateCurrentTime, 1000);

  initializeHeatmap();

  loadAllData();

  refreshInterval = setInterval(loadAllData, REFRESH_INTERVAL);
}

function setupEventListeners() {
  document.getElementById("refreshBtn").addEventListener("click", loadAllData);

  document.getElementById("timeRange").addEventListener("change", function () {
    loadStatistics();
  });

  document.getElementById("locationSelector").addEventListener("change", function () {
    selectedLocation = this.value;
    loadStatistics();
    updateHeatmap();
  });

  document.getElementById("predictionBtn").addEventListener("click", function () {
    openPredictionModal();
  });

  document.querySelector(".close-modal").addEventListener("click", function () {
    document.getElementById("predictionModal").style.display = "none";
  });

  window.addEventListener("click", function (event) {
    if (event.target === document.getElementById("predictionModal")) {
      document.getElementById("predictionModal").style.display = "none";
    }
  });

  document.getElementById("runPredictionBtn").addEventListener("click", function () {
    runPrediction();
  });
}

function updateCurrentTime() {
  const now = new Date();
  document.getElementById("currentTime").textContent = now.toLocaleString();
}

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

function processLocationData(records) {
  if (!records || records.length === 0) return;

  const uniqueLocations = [...new Set(records.map((record) => record.location))];

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

  locations = locationData;

  updateLocationList();

  updateLocationSelectors();
}

function updateLocationList() {
  const locationList = document.getElementById("locationList");

  locationList.innerHTML = "";

  if (locations.length === 0) {
    locationList.innerHTML = '<div class="no-data">暂无位置数据</div>';
    return;
  }

  locations.sort((a, b) => b.latestDensity - a.latestDensity);

  locations.forEach((location) => {
    const locationItem = document.createElement("div");
    locationItem.className = "location-item";
    if (selectedLocation === location.name) {
      locationItem.classList.add("active");
    }

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

    locationItem.addEventListener("click", function () {
      document.querySelectorAll(".location-item").forEach((item) => item.classList.remove("active"));
      this.classList.add("active");
      selectedLocation = location.name;

      document.getElementById("locationSelector").value = selectedLocation;

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
    const currentValue = selector.value;

    while (selector.options.length > 1) {
      selector.remove(1);
    }

    locations.forEach((location) => {
      const option = document.createElement("option");
      option.value = location.name;
      option.textContent = location.name;
      selector.appendChild(option);
    });

    if (currentValue && locations.some((loc) => loc.name === currentValue)) {
      selector.value = currentValue;
    }
  });
}

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

function updateStatisticsUI(statistics) {
  if (!statistics) return;

  document.getElementById("currentDensity").textContent = statistics.density.current.toFixed(2);
  document.getElementById("avgDensity").textContent = statistics.density.average.toFixed(2);
  document.getElementById("maxDensity").textContent = statistics.density.max.toFixed(2);

  document.getElementById("currentPeople").textContent = Math.round(statistics.people.current);

  const currentDensityElement = document.getElementById("currentDensity");
  if (statistics.density.current >= 1.0) {
    currentDensityElement.style.color = "#dc3545";
  } else if (statistics.density.current >= 0.5) {
    currentDensityElement.style.color = "#ffc107";
  } else {
    currentDensityElement.style.color = "#28a745";
  }
}

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

function updateHeatmap() {
  if (!locations || locations.length === 0) return;

  // 如果选择了特定位置，则只显示该位置
  let filteredLocations = locations;
  if (selectedLocation) {
    filteredLocations = locations.filter((loc) => loc.name === selectedLocation);
  }

  const container = document.getElementById("heatmapContainer");
  const width = container.clientWidth;
  const height = container.clientHeight;

  const points = [];
  const maxValue = Math.max(...filteredLocations.map((loc) => loc.latestDensity));

  filteredLocations.forEach((location, index) => {
    const x = Math.round(((index + 1) * width) / (filteredLocations.length + 1));
    const y = Math.round(height / 2 + (Math.random() - 0.5) * height * 0.6);

    points.push({
      x: x,
      y: y,
      value: location.latestDensity,
      radius: 30 + location.latestDensity * 20,
    });

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

    const oldLabels = container.querySelectorAll(".location-label");
    oldLabels.forEach((label) => label.remove());

    container.appendChild(label);
  });

  heatmapInstance.setData({
    max: Math.max(1.5, maxValue),
    data: points,
  });
}

function openPredictionModal() {
  document.getElementById("predictionModal").style.display = "block";

  if (selectedLocation) {
    document.getElementById("predictionLocation").value = selectedLocation;
  }

  runPrediction();
}

async function runPrediction() {
  try {
    const location = document.getElementById("predictionLocation").value;
    const hours = document.getElementById("predictionHours").value;

    const chartContainer = document.querySelector(".prediction-chart-container");
    chartContainer.innerHTML = `
            <div class="loading-spinner">
                <i class="fas fa-circle-notch fa-spin"></i>
                <span>正在生成预测...</span>
            </div>
        `;

    const requestData = {
      location: location,
      hours: parseInt(hours),
    };

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

    chartContainer.innerHTML = '<canvas id="predictionChart"></canvas>';

    displayPredictionChart(result.prediction);
  } catch (error) {
    console.error("预测失败:", error);
    showToast("生成预测失败，请稍后重试", "error");

    const chartContainer = document.querySelector(".prediction-chart-container");
    chartContainer.innerHTML = `
            <div class="prediction-error">
                <p><i class="fas fa-exclamation-triangle"></i> 预测失败: ${error.message}</p>
                <p>请确保有足够的历史数据用于预测</p>
            </div>
        `;
  }
}

function displayPredictionChart(prediction) {
  if (!prediction || !prediction.predictions || prediction.predictions.length === 0) {
    return;
  }

  const labels = prediction.predictions.map((p) => {
    const date = new Date(p.timestamp);
    return `${date.getHours()}:00`;
  });

  const densityData = prediction.predictions.map((p) => p.predicted_density);

  const timeOfDayColors = prediction.predictions.map((p) => {
    const hour = new Date(p.timestamp).getHours();
    if (hour >= 6 && hour < 12) return "rgba(255, 193, 7, 0.8)";
    if (hour >= 12 && hour < 18) return "rgba(0, 123, 255, 0.8)";
    return "rgba(108, 117, 125, 0.8)";
  });

  if (predictionChart) {
    predictionChart.destroy();
  }

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

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.style.backgroundColor = type === "error" ? "#f8d7da" : "#d1e7dd";
  toast.style.color = type === "error" ? "#842029" : "#0f5132";
  toast.style.padding = "1rem";
  toast.style.marginBottom = "10px";
  toast.style.borderRadius = "4px";
  toast.style.boxShadow = "0 2px 5px rgba(0,0,0,0.2)";
  toast.style.minWidth = "250px";

  const icon = document.createElement("i");
  icon.className = type === "error" ? "fas fa-exclamation-circle" : "fas fa-info-circle";
  icon.style.marginRight = "10px";
  toast.appendChild(icon);

  const messageText = document.createTextNode(message);
  toast.appendChild(messageText);

  toastContainer.appendChild(toast);

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

function simulateRealTimeData() {
  if (!locations || locations.length === 0) return;

  locations.forEach((location) => {
    const changeRate = 1 + (Math.random() - 0.5) * 0.1;

    location.latestDensity *= changeRate;

    location.latestDensity = Math.max(0.05, Math.min(2.0, location.latestDensity));

    const area = location.records[0].area_size;
    location.latestPeople = location.latestDensity * area;
  });

  updateLocationList();
  updateHeatmap();

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
