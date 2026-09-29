# 📡 Secondary Radar Data Analysis Tool

## About the Project
**Secondary Radar Data Analysis Tool** turns raw, noisy secondary radar data into clear, easy-to-read 3D flight paths. Radar captures (PCAP files) are converted to CSV, cleaned, and filtered to remove false detections. The tool then shows every aircraft on an interactive 3D radar screen and predicts where each one came from and where it is heading. The tool is built as a two-phase pipeline: first classify the hits, then model the trajectories.

## Features
- 🧹 **Noise filtering**: removes false sidelobe detections using the ACP (Azimuth Change Pulse) spread of each target
- 🎯 **Target classification**: DBSCAN clustering separates real targets (Main Lobe) from small clusters and noise
- 📡 **Radar PPI view**: interactive polar plot (Plan Position Indicator) with hover details for range, azimuth and cluster ID
- 🛩️ **3D aircraft tracks**: every aircraft is plotted in 3D using range, azimuth and height, and coloured by its Mode 3/A code
- 🔮 **Trajectory prediction**: spline curves smooth each flight path and extend it forward to predict where the aircraft is going
- 🎛️ **Dropdown selector**: view all aircraft together or focus on a single aircraft
- 📊 **Accuracy check**: the report validates the model with an average path error of about **0.65 km**
- 🌐 **Shareable output**: results export as a standalone HTML file that opens in any browser

## Tech Stack

| Area | Tools |
|---|---|
| **Language** | Python |
| **Data handling** | Pandas, NumPy |
| **Clustering** | scikit-learn (DBSCAN) |
| **Trajectory modelling** | SciPy (`splprep`, `splev` B-spline fitting) |
| **Visualization** | Plotly (`Scatterpolar`, `Scatter3d`, `Surface`) |
| **Data capture** | Wireshark (PCAP files) |
| **Data format** | CSV, ASTERIX CAT048 |
| **Output** | Interactive HTML |

## How It's Built
The project has two phases:

**Phase 1: Target classification** (`classification_of_hits/`)
1. Load radar hits and clean the data.
2. Group hits by target and range, and measure the ACP spread. A small spread means a **mainlobe** hit; a large spread means a **sidelobe** hit, which is rejected.
3. Convert range and azimuth to x/y coordinates, then run **DBSCAN**.
4. Label clusters as **Main Lobe**, **Small Cluster** or **Noise**, and plot them on a radar PPI.

**Phase 2: Trajectory analysis** (`trajectory analysis/`)
1. Load the track data (packet time, height, range, azimuth, Mode 3/A code).
2. Convert polar coordinates to 3D Cartesian (x, y, z in km).
3. Fit a **B-spline** to each aircraft's path with SciPy, then extend it by 30% to predict where it is heading.
4. Draw everything in one interactive **3D Plotly** view with a dropdown per aircraft.

## Project Structure
```
├── classification_of_hits/
│   ├── test1.py              # Mainlobe filtering + DBSCAN + PPI plot
│   └── csvfile.csv           # Sample radar hit data
├── trajectory analysis/
│   ├── test5.py              # 3D tracks + spline prediction
│   ├── TrackCSVFile.csv      # Track data
│   ├── landing.csv           # Additional radar data
│   └── ppi_3d_ai_trajectory.html   # Sample output
├── Minor_Project_Report_radar_final.docx
└── SEMINAR_ppt_radar.pptx
```

## Run It Locally
1. Install Python, then install the packages:
```bash
   pip install pandas numpy scikit-learn scipy plotly
```
2. Classify the radar hits:
```bash
   cd classification_of_hits
   python test1.py
```
3. Build the 3D trajectory view:
```bash
   cd "trajectory analysis"
   python test5.py
```
4. Both scripts open the result in your browser.

## Author
**Shivam Biala**
