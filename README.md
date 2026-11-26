# 🚦 TrafficVision
### AI-Based Traffic Monitoring, Congestion Analysis & Traffic Advisory System

TrafficVision is an intelligent traffic monitoring system that uses computer vision and AI-based techniques to analyze traffic videos, detect and track vehicles, estimate traffic congestion levels, and provide traffic management recommendations.

The system processes uploaded traffic videos and presents the results through an interactive Streamlit dashboard, helping users understand traffic conditions through visual analytics and actionable insights.

## 🌟 Key Features

### 1. Vehicle Detection
- Detects vehicles from uploaded traffic videos using the YOLO11n object detection model.
- Identifies four vehicle categories:
  - Cars
  - Motorcycles
  - Buses
  - Trucks

### 2. Vehicle Tracking & Counting
- Tracks detected vehicles across video frames.
- Assigns unique IDs to tracked vehicles.
- Calculates the total number of unique vehicles detected.
- Displays vehicle counts by category.

### 3. Traffic Congestion Analysis
- Analyzes vehicle detection data to estimate traffic conditions.
- Classifies traffic into three levels:
  - 🟢 LOW
  - 🟡 MODERATE
  - 🔴 HIGH
- Calculates average vehicle detections per frame as a traffic indicator.

### 4. Traffic Advisory System
- Generates traffic recommendations based on the detected congestion level.
- Provides actionable suggestions to help understand and respond to traffic conditions.

### 5. Interactive Dashboard
- Built using Streamlit.
- Supports traffic video uploads and processing.
- Displays traffic analysis results and vehicle statistics.
- Shows annotated output videos with vehicle detections and tracking IDs.
- Provides processed video downloads.
- Presents traffic information through visual analytics.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLO11n | Vehicle detection |
| OpenCV | Video processing and image operations |
| Streamlit | Interactive dashboard |
| NumPy | Numerical operations |
| Pandas | Data handling and analysis |
| Object Tracking | Vehicle tracking and unique counting |

## 🏗️ Project Workflow

1. Upload a traffic video through the dashboard.
2. The video is processed frame by frame.
3. YOLO11n detects vehicles in each frame.
4. The tracking module associates detections across frames.
5. Unique vehicle counts are calculated by category.
6. Traffic analysis estimates the congestion level.
7. The advisory module generates recommendations.
8. Results and annotated videos are displayed on the dashboard.

## 📁 Project Structure

```text
TrafficVision/
│
├── app.py
├── requirements.txt
│
├── vehicle_detection/
│   ├── ...
│
├── vehicle_counting/
│   ├── ...
│
├── traffic_advisory/
│   ├── ...
│
├── traffic_analysis/
│   ├── ...
│
└── README.md
```

*Note: The structure above illustrates the main modules. Refer to the actual repository for the complete file structure.*

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or compatible version supported by the project dependencies.
- Git
- Visual Studio Code (recommended)

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/TrafficVision.git
```

### Step 2: Navigate to the project directory

```bash
cd TrafficVision
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the virtual environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Expected Output

TrafficVision provides:

- Annotated traffic videos with detected vehicles and tracking IDs.
- Total unique vehicle count.
- Category-wise vehicle statistics.
- Average vehicle detections per frame.
- Traffic congestion classification.
- Traffic management recommendations.
- Downloadable processed video output.

## 🎯 Project Objectives

- Automate vehicle detection and counting using computer vision.
- Analyze traffic conditions from video input.
- Identify different vehicle categories.
- Estimate congestion levels using vehicle detection data.
- Generate recommendations based on traffic conditions.
- Present traffic insights through an interactive dashboard.

## ⚠️ Limitations

- The current system analyzes uploaded video files rather than live CCTV streams.
- Traffic congestion classification is based on detection statistics and is an estimate, not a calibrated measurement of actual road density.
- Results may vary depending on video quality, lighting, camera angle, and vehicle visibility.
- The system does not currently provide verified GPS-based traffic information.

## 🔮 Future Enhancements

- Live CCTV traffic monitoring.
- Real-time traffic congestion alerts.
- Lane-wise traffic analysis.
- Traffic volume forecasting using historical data.
- Geographic traffic visualization using real location data.
- Integration with advanced AI-based traffic advisory services.

## 👥 Team Project

**TrafficVision – Intelligent Traffic Monitoring, Congestion Analysis & AI-Based Traffic Advisory System**

Developed as a collaborative project focused on computer vision, vehicle tracking, traffic analytics, and intelligent dashboard development.

---

**TrafficVision – Making Traffic Monitoring Smarter with AI.** 🚦
