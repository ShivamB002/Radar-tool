import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import numpy as np
from scipy.interpolate import splprep, splev
import time

# Radar location (New Delhi)
RADAR_LAT = 28.6139
RADAR_LON = 77.2090

# Load and clean CSV
df = pd.read_csv("TrackCSVFile.csv")
df.columns = df.columns.str.strip()



df.rename(columns={
    'Pkt Time': 'Pkt_Time',
    'Azimuth': 'Azimuth',
    'range': 'Range_km',
    'Height': 'Height_m',
    'PRT Number': 'PRT_Number'
}, inplace=True)

df['Range_km'] = pd.to_numeric(df['Range_km'], errors='coerce')
df['Azimuth'] = pd.to_numeric(df['Azimuth'], errors='coerce')
df['Height_m'] = pd.to_numeric(df['Height_m'], errors='coerce')
df['PRT_Number'] = df['PRT_Number'].astype(str)
df['M3Acode'] = pd.to_numeric(df['M3Acode'], errors='coerce')
df.dropna(subset=['Range_km', 'Azimuth', 'Height_m', 'M3Acode'], inplace=True)

df['Azimuth_rad'] = np.radians(df['Azimuth'])

# Convert to Cartesian coordinates
df['x'] = df['Range_km'] * np.sin(df['Azimuth_rad'])
df['y'] = df['Range_km'] * np.cos(df['Azimuth_rad'])
df['z'] = df['Height_m'] / 1000.0  # Convert to km

# Assign colors to M3Acodes
unique_m3a = sorted(df['M3Acode'].dropna().unique())
colors = px.colors.qualitative.Light24 + px.colors.qualitative.Dark24
color_map = {m3a: colors[i % len(colors)] for i, m3a in enumerate(unique_m3a)}
df['Color'] = df['M3Acode'].map(color_map)

traces = []
raw_data_traces_indices = {}
smoothed_trajectory_traces_indices = {}

# 1. Raw Aircraft Trajectories
for m3a in unique_m3a:
    sub_df = df[df['M3Acode'] == m3a].sort_values(by='Pkt_Time')
    trace = go.Scatter3d(
        x=sub_df['x'], y=sub_df['y'], z=sub_df['z'],
        mode='markers',
        marker=dict(size=4, color=color_map[m3a]),
        line=dict(color=color_map[m3a], width=2),
        name=f"M3Acode {int(m3a)}",
        text=[
            f"PRT: {prt}<br>Azimuth: {az}°<br>Range: {r} km<br>Height: {h:.2f} km<br>M3Acode: {fz}"
            for prt, az, r, h, fz in zip(sub_df['PRT_Number'], sub_df['Azimuth'], sub_df['Range_km'], sub_df['z'], sub_df['M3Acode'])
        ],
        hoverinfo='text',
        visible=True
    )
    traces.append(trace)
    raw_data_traces_indices[m3a] = len(traces) - 1

# 2. Radar Center Marker
radar_trace_index = len(traces)
traces.append(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode='markers+text',
    marker=dict(size=6, color='red'),
    text=["Radar"],
    textposition='top center',
    name="Radar",
    visible=True
))

# 3. Polar grid circles
circle_radii = [50, 100, 150, 200]
polar_grid_traces_start_index = len(traces)
for r in circle_radii:
    theta = np.linspace(0, 2 * np.pi, 200)
    traces.append(go.Scatter3d(
        x=r * np.sin(theta), y=r * np.cos(theta), z=np.zeros_like(theta),
        mode='lines',
        line=dict(color='green', width=3),
        name=f'Range {r} km',
        showlegend=False,
        hoverinfo='skip',
        visible=True
    ))
polar_grid_traces_end_index = len(traces) - 1

# 4. Radial lines every 30°
radial_lines_traces_start_index = len(traces)
for angle_deg in range(0, 360, 30):
    angle_rad = np.radians(angle_deg)
    traces.append(go.Scatter3d(
        x=[0, 200 * np.sin(angle_rad)],
        y=[0, 200 * np.cos(angle_rad)],
        z=[0, 0],
        mode='lines',
        line=dict(color='white', width=2),
        name=f'{angle_deg}°',
        showlegend=False,
        hoverinfo='skip',
        visible=True
    ))
radial_lines_traces_end_index = len(traces) - 1

# 5. Transparent ground plane
ground_plane_trace_index = len(traces)
plane_x, plane_y = np.meshgrid(np.linspace(-200, 200, 2), np.linspace(-200, 200, 2))
plane_z = np.zeros_like(plane_x)
traces.append(go.Surface(
    x=plane_x,
    y=plane_y,
    z=plane_z,
    showscale=False,
    opacity=0.05,
    colorscale=[[0, 'black'], [1, 'black']],
    hoverinfo='skip',
    name='Ground',
    visible=True
))



# 6. AI Smoothed Trajectories (with Extrapolation) 
for m3a in unique_m3a:
    sub_df = df[df['M3Acode'] == m3a].sort_values(by='Pkt_Time')

    if len(sub_df) >= 4:
        x, y, z = sub_df['x'].values, sub_df['y'].values, sub_df['z'].values
        coords = np.array(list(zip(x, y, z)))
        coords_unique = np.unique(coords, axis=0)

        if coords_unique.shape[0] >= 4:
            x, y, z = coords_unique[:, 0], coords_unique[:, 1], coords_unique[:, 2]
            try:
                #Capture the 'u' parameter values ---
                tck, u = splprep([x, y, z], s=0.5)

                # This will predict 30% beyond the last data point. Adjust 1.3 as needed.
                prediction_factor = 1.3
                u_new = np.linspace(u.min(), u.max() * prediction_factor, 200)
                
                # Evaluate the spline over the new, extended range
                new_points = splev(u_new, tck)
                
                #  Split for visualization 
                # Find where the extrapolation begins to plot it differently
                original_data_end_idx = np.where(u_new > u.max())[0][0]
                
                # Trace for the interpolated (known) path
                traces.append(go.Scatter3d(
                    x=new_points[0][:original_data_end_idx+1],
                    y=new_points[1][:original_data_end_idx+1],
                    z=new_points[2][:original_data_end_idx+1],
                    mode='lines',
                    line=dict(color='yellow', width=6),
                    name=f"Smoothed Path for {int(m3a)}",
                    hoverinfo='skip',
                    visible=False
                ))
                smoothed_trajectory_traces_indices[m3a] = len(traces) - 1 
                
                # Trace for the extrapolated (predicted) path
                traces.append(go.Scatter3d(
                    x=new_points[0][original_data_end_idx:],
                    y=new_points[1][original_data_end_idx:],
                    z=new_points[2][original_data_end_idx:],
                    mode='lines',
                    line=dict(color='cyan', width=6, dash='dot'), 
                    name=f"Extrapolated Path for {int(m3a)}",
                    hoverinfo='skip',
                    visible=False
                ))

            except Exception as e:
                print(f"[!] Skipped M3Acode {m3a}: spline fitting failed -> {e}")
                smoothed_trajectory_traces_indices[m3a] = None
        else:
            print(f"[!] Skipped M3Acode {m3a}: not enough unique points")
            smoothed_trajectory_traces_indices[m3a] = None
    else:
        print(f"[!] Skipped M3Acode {m3a}: less than 4 total points")
        smoothed_trajectory_traces_indices[m3a] = None

# --- Dropdown Buttons ---
dropdown_buttons = []

# Show All
show_all_visibility = [False] * len(traces)
for idx in raw_data_traces_indices.values():
    show_all_visibility[idx] = True
show_all_visibility[radar_trace_index] = True
for i in range(polar_grid_traces_start_index, polar_grid_traces_end_index + 1):
    show_all_visibility[i] = True
for i in range(radial_lines_traces_start_index, radial_lines_traces_end_index + 1):
    show_all_visibility[i] = True
show_all_visibility[ground_plane_trace_index] = True

dropdown_buttons.append(dict(
    label='Show All',
    method='update',
    args=[{'visible': show_all_visibility},
          {'title': '3D Radar PPI (All Aircraft - No Trajectories)'}]
))

# M3Acode-specific buttons
for m3a in unique_m3a:
    visibility = [False] * len(traces)

    raw_idx = raw_data_traces_indices.get(m3a)
    if raw_idx is not None:
        visibility[raw_idx] = True

    smoothed_idx = smoothed_trajectory_traces_indices.get(m3a)
    if smoothed_idx is not None:
        visibility[smoothed_idx] = True

    visibility[radar_trace_index] = True
    for i in range(polar_grid_traces_start_index, polar_grid_traces_end_index + 1):
        visibility[i] = True
    for i in range(radial_lines_traces_start_index, radial_lines_traces_end_index + 1):
        visibility[i] = True
    visibility[ground_plane_trace_index] = True

    dropdown_buttons.append(dict(
        label=f"M3Acode {int(m3a)}",
        method='update',
        args=[{'visible': visibility},
              {'title': f"Predicted Trajectory - M3Acode {int(m3a)}"}]
    ))

# Final figure setup
fig = go.Figure(data=traces)
fig.update_layout(
    title="3D Radar PPI (Polar Projection) with AI Trajectories",
    scene=dict(
        xaxis=dict(title='X (km)', range=[-200, 200], backgroundcolor='black', gridcolor='black'),
        yaxis=dict(title='Y (km)', range=[-200, 200], backgroundcolor='black', gridcolor='black'),
        zaxis=dict(title='Height (km)', range=[0, 12], backgroundcolor='black', gridcolor='black'),
        aspectmode='cube',
        bgcolor='black'
    ),
    scene_camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
    paper_bgcolor='black',
    font=dict(color='white'),
    margin=dict(l=0, r=0, t=40, b=0),
    updatemenus=[dict(
        buttons=dropdown_buttons,
        direction="down",
        showactive=True,
        x=1.2,
        y=1,
        xanchor="left",
        yanchor="top",
        bgcolor='black',
        bordercolor='gray',
        font=dict(color='white')
    )]
)

pio.write_html(fig, "ppi_3d_ai_trajectory.html", auto_open=True)
