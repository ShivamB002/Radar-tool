# #spicify mainlobe, noise and small cluster
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.cluster import DBSCAN

# === Step 1: Load and Prepare Data (Same as before) ===
try:
    df = pd.read_csv("csvfile.csv")
except FileNotFoundError:
    print("Warning: 'csvfile.csv' not found. Creating a dummy file for demonstration.")
    # Create dummy data with distinct mainlobe and sidelobe characteristics
    mainlobe_hits = 150
    large_cluster_hits = 80
    sidelobe_hits = 200
    data = {
        'ScanNo': np.ones(mainlobe_hits + large_cluster_hits + sidelobe_hits),
        'range': np.concatenate([
            np.random.normal(150000, 500, mainlobe_hits),      # A standard mainlobe cluster
            np.random.normal(250000, 800, large_cluster_hits), # A larger mainlobe cluster
            np.random.uniform(50000, 400000, sidelobe_hits)     # Sidelobes
        ]),
        'ACP': np.concatenate([
            np.random.randint(500, 508, mainlobe_hits),      # Narrow ACP spread
            np.random.randint(2000, 2010, large_cluster_hits), # Narrow ACP spread
            np.random.randint(0, 4095, sidelobe_hits)         # Wide ACP spread
        ]),
        'code': ['A'] * (mainlobe_hits + large_cluster_hits + sidelobe_hits)
    }
    df = pd.DataFrame(data)

df.columns = df.columns.str.strip()
df = df.dropna(subset=["range", "code", "ACP", "ScanNo"])
df['range'] = df['range'].astype(float)
df['ACP'] = df['ACP'].astype(int)
df['code'] = df['code'].astype(str)
df['ScanNo'] = df['ScanNo'].astype(int)

# === Step 2: Classify Hits (Same as before) ===
df['target_id'] = df['range'].round(-2).astype(str) + "_" + df['code']
acp_stats = df.groupby(['ScanNo', 'target_id'])['ACP'].agg(['min', 'max', 'count']).reset_index()
acp_stats['spread'] = acp_stats['max'] - acp_stats['min']
df = df.merge(acp_stats, on=['ScanNo', 'target_id'])
df['hit_type'] = df.apply(lambda row: 'mainlobe' if row['spread'] <= 15 else 'sidelobe', axis=1)


# === NEW: Step 3: Filter for Mainlobe Hits Only ===
print(f"Original total hits: {len(df)}")
mainlobe_df = df[df['hit_type'] == 'mainlobe'].copy()
print(f"Filtered down to {len(mainlobe_df)} mainlobe hits.\n")

#Step 4: Scale and Prepare Data for Clustering
# Filter out any remaining extreme values and scale to km
mainlobe_df = mainlobe_df[mainlobe_df['range'] < 500000]
mainlobe_df['range'] = mainlobe_df['range'] / 1000.0 # Convert meters to km
mainlobe_df['azimuth_deg'] = (mainlobe_df['ACP'] * 360) / 4096

# Step 5: Perform DBSCAN on Mainlobe Hits
print("Performing DBSCAN clustering on mainlobe hits...")
theta_rad = np.deg2rad(mainlobe_df['azimuth_deg'])
mainlobe_df['x'] = mainlobe_df['range'] * np.cos(theta_rad)
mainlobe_df['y'] = mainlobe_df['range'] * np.sin(theta_rad)
X = mainlobe_df[['x', 'y']].values

db = DBSCAN(eps=10, min_samples=10).fit(X)
mainlobe_df['cluster'] = db.labels_

# NEW: Step 6: Identify Large Clusters as "Main Lobes" 
# Define what constitutes a "large" cluster (e.g., more than 50 hits)
CLUSTER_SIZE_THRESHOLD = 50
print(f"Identifying main lobes as clusters with > {CLUSTER_SIZE_THRESHOLD} hits...")

# Calculate the size of each cluster
cluster_counts = mainlobe_df['cluster'].value_counts()
# Get the IDs of the large clusters (ignoring noise cluster -1)
large_cluster_ids = cluster_counts[cluster_counts > CLUSTER_SIZE_THRESHOLD].index
large_cluster_ids = large_cluster_ids[large_cluster_ids != -1]

# Classify each point based on its cluster size
def classify_cluster_type(cluster_id):
    if cluster_id in large_cluster_ids:
        return 'Main Lobe'
    elif cluster_id == -1:
        return 'Noise'
    else:
        return 'Small Cluster'

mainlobe_df['cluster_type'] = mainlobe_df['cluster'].apply(classify_cluster_type)

print(f"Found {len(large_cluster_ids)} significant Main Lobe(s).")

#Step 7: Plot the Results
fig = go.Figure()

# Define colors for each category
color_map = {
    'Main Lobe': 'red',
    'Small Cluster': 'green',
    'Noise': '#cccccc'  # light gray
}

# Add a separate trace for each category to create a legend
for cluster_type, group_df in mainlobe_df.groupby('cluster_type'):
    fig.add_trace(go.Scatterpolar(
        r=group_df['range'],
        theta=group_df['azimuth_deg'],
        mode='markers',
        marker=dict(
            color=color_map[cluster_type],
            size=6
        ),
        name=cluster_type,  # This name will appear in the legend
        hovertemplate=f'<b>{cluster_type}</b><br>' +
                      '<b>Range</b>: %{r:.2f} km<br>' +
                      '<b>Azimuth</b>: %{theta:.2f}°<br>' +
                      '<b>Cluster ID</b>: ' + group_df['cluster'].astype(str)
    ))

# Update layout
fig.update_layout(
    title="Radar PPI: Main Lobe Clusters by Size",
    polar=dict(
        radialaxis=dict(title="Range (km)", visible=True),
        angularaxis=dict(direction="clockwise", rotation=90)
    ),
    legend_title_text='Detection Type'
)

pio.renderers.default = "browser"
fig.show()