import streamlit as st
import io
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from math import degrees
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import math
import datetime

# --- Helper: Bearing ---
def get_bearing(lat1, lon1, lat2, lon2):
    dlon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    angle = np.arctan2(x, y)
    return (degrees(angle) + 360) % 360

# --- Feature extraction for HYSPLIT ---
def extract_features_hysplit(df):
    features = []
    for traj_id, group in df.groupby("traj_id"):
        group = group.sort_values("hour_along")
        start, end = group.iloc[0], group.iloc[-1]
        distance = geodesic((start["lat"], start["lon"]), (end["lat"], end["lon"])).km
        angle = get_bearing(start["lat"], start["lon"], end["lat"], end["lon"])
        features.append({
            "traj_id": traj_id,
            "distance": distance,
            "angle": angle
        })
    return pd.DataFrame(features)

# --- Prepare trajectory array for DTW clustering ---
def prepare_trajectory_array(df):
    traj_ids = df['traj_id'].drop_duplicates().values.tolist()
    traj_list = []
    max_len = df.groupby('traj_id').size().max()
    for tid in traj_ids:
        sub = df[df['traj_id']==tid].sort_values('hour_along')[['lat', 'lon']].values
        pad = np.full((max_len - len(sub), 2), np.nan)
        arr = np.vstack([sub, pad])
        # Forward fill NaNs with last valid value
        mask_nan = np.isnan(arr[:,0])
        if mask_nan.any():
            arr[mask_nan, :] = arr[~mask_nan, :][-1]
        traj_list.append(arr)
    return np.array(traj_list), traj_ids
def cluster_trajectories_Kmeans(df, n_clusters=4, random_state=50):
    """
    Cluster trajectories using KMeans on padded lat/lon sequences.

    Parameters:
        df (pd.DataFrame): Must contain columns 'trajectory_id', 'lat', 'lon'
        n_clusters (int): Number of clusters
        random_state (int): Random seed for reproducibility

    Returns:
        labels (np.ndarray): Cluster assignments for each trajectory (order matches trajectory_ids)
        trajectory_ids (np.ndarray): Array of trajectory IDs (order matches labels)
        kmeans (KMeans): Fitted KMeans model
    """
    trajectory_ids = df['traj_id'].unique()
    max_len = df.groupby('traj_id').size().max()
    features = []

    # Step 1: Create feature vectors from lat/lon, pad with NaN then replace with 0
    for traj_id in trajectory_ids:
        traj_df = df[df['traj_id'] == traj_id]
        coords = np.column_stack((traj_df['lon'].values, traj_df['lat'].values)).flatten()
        coords = np.pad(coords, (0, max_len*2 - len(coords)), constant_values=np.nan)
        features.append(coords)

    features = np.nan_to_num(features)  # replace NaNs with 0

    # Step 2: Cluster using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(features)

    return labels, trajectory_ids, kmeans
# --- Clustering Methods ---
def cluster_kmeans(features_df, n_clusters):
    features_df["x"] = np.cos(np.radians(features_df["angle"]))
    features_df["y"] = np.sin(np.radians(features_df["angle"]))
    X = StandardScaler().fit_transform(features_df[["x", "y", "distance"]])
    km = KMeans(n_clusters=n_clusters, random_state=42)
    features_df["cluster"] = km.fit_predict(X)
    return features_df

def cluster_dtw_kmeans(X, n_clusters):
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
    labels = model.fit_predict(X)
    return labels

def cluster_agglomerative_dtw(X, n_clusters):
    D = cdist_dtw(X)
    model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    labels = model.fit_predict(D)
    return labels

def cluster_softdtw_kmeans(X, n_clusters):
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="softdtw", random_state=42)
    labels = model.fit_predict(X)
    return labels

# --- Plot polar cluster output ---
def plot_clusters(features_df):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    cmap = plt.get_cmap("tab10")
    for label, group in features_df.groupby("cluster"):
        theta = np.radians(group["angle"])
        r = group["distance"]
        ax.scatter(theta, r, label=f"Cluster {label}", alpha=0.75, s=40, color=cmap(label))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Trajectory Clusters (Wind Rose)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
    return fig

# --- Plot all trajectories by cluster on basemap ---
def plot_clusters_with_basemap_single(df, labels, n_clusters, title_prefix, traj_ids):
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'olive', 'gray']
    all_lons = df['lon'].values
    all_lats = df['lat'].values
    margin = 1.0
    min_lon, max_lon = all_lons.min() - margin, all_lons.max() + margin
    min_lat, max_lat = all_lats.min() - margin, all_lats.max() + margin

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.5)

    for i in range(n_clusters):
        cluster_indices = [j for j, lab in enumerate(labels) if lab == i]
        for idx in cluster_indices:
            tid = traj_ids[idx]
            sub = df[df['traj_id']==tid].sort_values('hour_along')[['lat', 'lon']].values
            if len(sub) > 1:
                ax.plot(sub[:,1], sub[:,0], color=colors[i % len(colors)], alpha=0.5)
    ax.set_title(f"{title_prefix} - All Trajectories by Cluster")
    return fig

import math

def plot_clusters_with_basemapold(df, labels, n_clusters, title_prefix):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    import pandas as pd

    trajectory_ids = df['trajectory_id'].unique()
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'olive', 'gray']
    all_lons = df['lon'].values
    all_lats = df['lat'].values
    margin = 1.0
    min_lon, max_lon = all_lons.min() - margin, all_lons.max() + margin
    min_lat, max_lat = all_lats.min() - margin, all_lats.max() + margin

    nrows = 2
    ncols = math.ceil(n_clusters / 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 7),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    for i in range(n_clusters):
        ax = axes[i]
        ax.set_title(f'{title_prefix} Cluster {i+1}')
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        ax.stock_img()
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.5)
        for j, traj_id in enumerate(trajectory_ids):
            if labels[j] == i:
                traj_df = df[df['trajectory_id'] == traj_id]
                ax.plot(traj_df['lon'], traj_df['lat'], color=colors[i % len(colors)], alpha=0.5, transform=ccrs.PlateCarree())
                ax.plot(traj_df['lon'].values[-1], traj_df['lat'].values[-1],
                        marker='o', markersize=6, color='black',
                        markeredgecolor='yellow', markeredgewidth=1.5,
                        linestyle='None', zorder=5, transform=ccrs.PlateCarree())

    # Show cluster fractions
    cluster_counts = pd.Series(labels).value_counts(normalize=True).sort_index()
    for i, ax in enumerate(axes[:n_clusters]):
        fraction = cluster_counts.get(i, 0)
        ax.text(0.05, 0.95, f'Fraction: {fraction:.2%}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Hide unused axes if n_clusters is odd
    for ax in axes[n_clusters:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig

def plot_clusters_with_basemap(df, labels, n_clusters, title_prefix, traj_ids):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'olive', 'gray']
    all_lons = df['lon'].values
    all_lats = df['lat'].values
    margin = 1.0
    min_lon, max_lon = all_lons.min() - margin, all_lons.max() + margin
    min_lat, max_lat = all_lats.min() - margin, all_lats.max() + margin

    nrows = 2
    ncols = math.ceil(n_clusters / 2)
    fig, axs = plt.subplots(nrows, ncols, figsize=(7*ncols, 10), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()  # Flatten to 1D for easy iteration

    for i in range(n_clusters):
        ax = axs[i]
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        ax.stock_img()
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.5)
        cluster_indices = [j for j, lab in enumerate(labels) if lab == i]
        for idx in cluster_indices:
            tid = traj_ids[idx]
            sub = df[df['traj_id']==tid].sort_values('hour_along')[['lat', 'lon']].values
            if len(sub) > 1:
                ax.plot(sub[:,1], sub[:,0], color=colors[i % len(colors)], alpha=0.5)
                #ax.plot(traj_df['lon'], traj_df['lat'], color=colors[i], alpha=0.5)
                ax.plot(sub[:,1], sub[:,0],
                        marker='o', markersize=3, color=colors[i % len(colors)],
                        markeredgecolor='yellow', markeredgewidth=.5,
                        linestyle='None', zorder=5)
                ax.plot(sub[:,1][-1], sub[:,0][-1],
                        marker='o', markersize=6, color='black',
                        markeredgecolor='yellow', markeredgewidth=.5,
                        linestyle='None', zorder=5)
                
        ax.set_title(f"{title_prefix} - Cluster {i+1}")

    # Show cluster fractions
    cluster_counts = pd.Series(labels).value_counts(normalize=True).sort_index()
    for i, ax in enumerate(axs[:n_clusters]):
        fraction = cluster_counts.get(i, 0)
        ax.text(0.05, 0.95, f'Fraction: {fraction:.2%}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # Hide unused subplots if n_clusters is odd
    for j in range(n_clusters, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    return fig

# --- Plot mean trajectories on basemap ---
def plot_mean_trajectories(df, labels, traj_ids, n_clusters):
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'olive', 'gray']
    all_lons = df['lon'].values
    all_lats = df['lat'].values
    margin = 1.0
    min_lon, max_lon = all_lons.min() - margin, all_lons.max() + margin
    min_lat, max_lat = all_lats.min() - margin, all_lats.max() + margin

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.5)

    for i in range(n_clusters):
        cluster_indices = [j for j, lab in enumerate(labels) if lab == i]
        cluster_trajs = []
        for idx in cluster_indices:
            tid = traj_ids[idx]
            sub = df[df['traj_id']==tid].sort_values('hour_along')[['lat', 'lon']].values
            cluster_trajs.append(sub)
        # Pad and mean
        if cluster_trajs:
            max_len = max(len(t) for t in cluster_trajs)
            padded = []
            for t in cluster_trajs:
                pad = np.full((max_len - len(t), 2), np.nan)
                arr = np.vstack([t, pad])
                mask_nan = np.isnan(arr[:,0])
                if mask_nan.any():
                    arr[mask_nan, :] = arr[~mask_nan, :][-1]
                padded.append(arr)
            mean_traj = np.nanmean(np.stack(padded), axis=0)
            ax.plot(mean_traj[:,1], mean_traj[:,0], color=colors[i % len(colors)], linewidth=3, label=f'Cluster {i+1}')
    ax.legend()
    ax.set_title("Mean Trajectory per Cluster")
    return fig

# --- Streamlit App UI ---
st.set_page_config("HYSPLIT Clustering", layout="wide")
st.title("🌬️ HYSPLIT Trajectory Cluster Explorer")

uploaded_file = st.file_uploader("Upload HYSPLIT trajectory CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    expected_cols = {"site_name", "run", "receptor", "hour_along", "lat", "lon", "height_i", "traj_dt_i"}
    if not expected_cols.issubset(df.columns):
        st.error(f"Missing columns. Required: {expected_cols}")
    else:
        # Convert traj_dt_i to date
        df["traj_dt_i"] = pd.to_datetime(df["traj_dt_i"], errors='coerce').dt.date
        # Create traj_id by combining site_name and traj_dt_i
        df["traj_id"] = df["site_name"].astype(str) + "_" + df["traj_dt_i"].astype(str)

        # Sidebar: Date range selection
        min_date = df["traj_dt_i"].min()
        max_date = df["traj_dt_i"].max()
        date_range = st.sidebar.date_input("Select date range for traj_dt_i", [min_date, max_date], min_value=min_date, max_value=max_date)
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df["traj_dt_i"] >= start_date) & (df["traj_dt_i"] <= end_date)]

        # Sidebar: Select site_name, run, and initial height
        site_names = sorted(df['site_name'].unique())
        site = st.sidebar.selectbox("Select site name", site_names)
        runs = sorted(df[df['site_name']==site]['run'].unique(),reverse=True)
        run = st.sidebar.selectbox("Select run", runs)
        heights = sorted(df[(df['site_name']==site) & (df['run']==run)]['height_i'].unique())
        height = st.sidebar.selectbox("Select initial height", heights)
        randomstate = st.sidebar.number_input("Random state", min_value=0, max_value=10000, value=50, step=1)#st.sidebar.slider("Random states", 0, 1000, 50)
        filtered_df = df[(df['site_name']==site) & (df['run']==run) & (df['height_i']==height)]

        n = st.sidebar.slider("Number of Clusters", 2, 10, 4)
        method = st.sidebar.selectbox("Clustering Method", [
            "Kmeans",
            "KMeans (distance+angle)", 
            "DTW KMeans", 
            "Agglomerative DTW", 
            "Soft-DTW KMeans"
        ])

        with st.spinner("Extracting features..."):
            features_df = extract_features_hysplit(filtered_df)
            #labels, trajectory_ids, kmeans_model = cluster_trajectories_Kmeans(filtered_df, n_clusters=n, random_state=randomstate)
            if method == "KMeans (distance+angle)":
                clustered_df = cluster_kmeans(features_df.copy(), n)
                st.subheader("📊 Clustered Trajectory Summary")
                st.dataframe(clustered_df)
                st.subheader("🧭 Wind Rose–Style Cluster Plot")
                st.pyplot(plot_clusters(clustered_df))
                st.subheader("🗺️ All Trajectories by Cluster (Basemap)")
                st.pyplot(plot_clusters_with_basemap(filtered_df, clustered_df["cluster"].values, n, method, clustered_df["traj_id"].tolist()))
                st.subheader("🗺️ Mean Trajectory per Cluster")
                st.pyplot(plot_mean_trajectories(filtered_df, clustered_df["cluster"].values, clustered_df["traj_id"].tolist(), n))
                fig = plot_clusters_with_basemap(filtered_df, clustered_df["cluster"].values, n, method, clustered_df["traj_id"].tolist())
            else:
                # Prepare padded arrays for DTW-based clustering
                X, traj_ids = prepare_trajectory_array(filtered_df)
                if method == "DTW KMeans":
                    labels = cluster_dtw_kmeans(X, n)
                elif method == "Agglomerative DTW":
                    labels = cluster_agglomerative_dtw(X, n)
                elif method == "Soft-DTW KMeans":
                    labels = cluster_softdtw_kmeans(X, n)
                elif method =="Kmeans":
                    labels, traj_ids, kmeans_model = cluster_trajectories_Kmeans(filtered_df, n_clusters=n, random_state=randomstate)
                features_df = extract_features_hysplit(filtered_df)
                features_df["cluster"] = labels
                #st.subheader("🧭 Wind Rose–Style Cluster Plot")
                #st.pyplot(plot_clusters(features_df))
                st.subheader("🗺️ All Trajectories by Cluster (Basemap)")
                st.pyplot(plot_clusters_with_basemap(filtered_df, labels, n, method, traj_ids))
                fig = plot_clusters_with_basemap(filtered_df, labels, n, method, traj_ids)
                st.subheader("🗺️ Mean Trajectory per Cluster")
                st.pyplot(plot_mean_trajectories(filtered_df, labels, traj_ids, n))
                fig2 = plot_mean_trajectories(filtered_df, labels, traj_ids, n)
                st.subheader("📊 Clustered Trajectory Summary")
                st.dataframe(features_df)

            st.download_button("📥 Download Cluster Results",
                               features_df.to_csv(index=False),
                               "clustered_trajectories.csv",
                               "text/csv")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cluster_{n}_{method}_{timestamp}.png"
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                label="Download indivials hysplits as clustered figure as PNG",
                data=buf,
                file_name=filename,
                mime="image/png")
            buf2 = io.BytesIO()
            fig.savefig(buf2, format="png", bbox_inches="tight")
            buf2.seek(0)
            st.download_button(
                label="Download mean hysplits figure as PNG",
                data=buf2,
                file_name=f"mean_{filename}",
                mime="image/png")