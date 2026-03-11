#%% IMPORTS
import os
import numpy as np
import rasterio
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# MAIN FUNCTIONS: image loading and savngs
def preprocess_image(filepath):
    with rasterio.open(filepath) as src:
        image = src.read()
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        image_shape = image.shape
        if image.ndim == 3:
            image_vector = image.reshape(image.shape[0], -1).T
        else:
            image_vector = image.flatten().reshape(-1, 1)

        meta = src.meta.copy()

    return image_vector, image_shape, meta

def save_predicted_cluster_matrix(predicted_cluster_matrix, output_dir, original_meta):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import os
    import rasterio

    # Copy and update metadata
    meta = original_meta.copy()
    meta.update(
        count=1,
        dtype=rasterio.int16,
        nodata=-999  # !!Use your preferred nodata value!!
    )

    cluster_data = predicted_cluster_matrix[np.newaxis, :, :].astype(np.int16)

    # Output paths
    predicted_tif_path = os.path.join(output_dir, "Predicted_Cluster_Matrix.tif")
    predicted_png_path = os.path.join(output_dir, "Predicted_Cluster_Matrix.png")

    # Save GeoTIFF
    with rasterio.open(predicted_tif_path, 'w', **meta) as dst:
        dst.write(cluster_data)

    # Plot as PNG
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_values = np.unique(predicted_cluster_matrix[~np.isnan(predicted_cluster_matrix)])
    cmap = plt.get_cmap('turbo', len(unique_values))
    norm = colors.BoundaryNorm(np.arange(len(unique_values) + 1) - 0.5, cmap.N)
    cax = ax.imshow(predicted_cluster_matrix, cmap=cmap, norm=norm)
    colorbar = fig.colorbar(cax, ax=ax, ticks=np.arange(len(unique_values)))
    colorbar.ax.set_yticklabels([int(v) for v in unique_values])
    ax.set_title('Predicted Cluster Matrix')
    ax.axis('off')
    fig.savefig(predicted_png_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f" Saved georeferenced TIFF to: {predicted_tif_path}")
    print(f"️ Saved PNG plot to: {predicted_png_path}")

def plot_feature_distributions_by_cluster(valid_image_2d, cluster_indices, output_dir, feature_names=None, sample_fraction=0.05):
    print("Generating feature-wise violin plots by cluster...")

    if sample_fraction < 1.0:
        np.random.seed(42)
        sample_size = int(len(valid_image_2d) * sample_fraction)
        indices = np.random.choice(len(valid_image_2d), sample_size, replace=False)
        data_sample = valid_image_2d[indices]
        cluster_sample = cluster_indices[indices]
    else:
        data_sample = valid_image_2d
        cluster_sample = cluster_indices

    # Clip to 1st–99th percentile to reduce skew
    p1 = np.percentile(data_sample, 1, axis=0)
    p99 = np.percentile(data_sample, 99, axis=0)
    data_sample = np.clip(data_sample, p1, p99)

    num_features = data_sample.shape[1]
    if feature_names is None:
        feature_names = [f'Band {i+1}' for i in range(num_features)]

    violin_dir = os.path.join(output_dir, "Feature_Violin_Plots")
    os.makedirs(violin_dir, exist_ok=True)

    for i in range(num_features):
        values = data_sample[:, i]
        df = pd.DataFrame({
            'Value': values,
            'Cluster': cluster_sample
        })

        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Cluster', y='Value', hue='Cluster', data=df,
                       palette='husl', inner='box', cut=0, legend=False)

        plt.title(f'Feature Distribution by Cluster - {feature_names[i]}')
        plt.xlabel('Cluster')
        plt.ylabel('Value')
        plt.grid(True)

        output_path = os.path.join(violin_dir, f'Feature_{i+1}_{feature_names[i].replace(" ", "_")}_Violin.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved violin plot for {feature_names[i]} to: {output_path}")

def plot_input_feature_correlation_matrix(valid_image_2d, output_dir, feature_names=None):
    """
    Plots the correlation matrix between input features before clustering.
    """
    print("Plotting input feature correlation matrix...")

    num_features = valid_image_2d.shape[1]
    if feature_names is None:
        feature_names = [f'Band {i + 1}' for i in range(num_features)]

    df = pd.DataFrame(valid_image_2d, columns=feature_names)
    corr_matrix = df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
                square=True, linewidths=0.5, annot_kws={"size": 7})
    plt.title("Input Feature Correlation Matrix", fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "Input_Feature_Correlation_Matrix.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Input correlation matrix saved to: {output_path}")

# PLOT INDIVIDUAL CLUSTERS
def plot_individual_clusters(filepath, predicted_cluster_matrix, mainDir):
    with rasterio.open(filepath) as src:
        meta = src.meta.copy()

    unique_clusters = np.unique(predicted_cluster_matrix[~np.isnan(predicted_cluster_matrix)])
    turbo_cmap = plt.cm.turbo
    colors = [turbo_cmap(i / len(unique_clusters)) for i in range(len(unique_clusters))]
    colors.insert(0, (1, 1, 1, 1))  # Adding white for NaN values
    new_cmap = ListedColormap(colors)

    for cluster_id in unique_clusters:
        fig, ax = plt.subplots(figsize=(10, 8))
        highlighted_image = np.zeros(predicted_cluster_matrix.shape, dtype=int)
        mask = predicted_cluster_matrix == cluster_id
        highlighted_image[mask] = int(np.where(unique_clusters == cluster_id)[0]) + 1

        cax = ax.imshow(highlighted_image, cmap=new_cmap, vmin=0, vmax=len(unique_clusters))
        ax.set_title(f'Cluster {int(cluster_id)} Isolated')
        ax.axis('off')

        png_output_file = f'{mainDir}/Isolated_Cluster_{int(cluster_id)}.png'
        fig.savefig(png_output_file, dpi=300)
        plt.close(fig)
        print(f"Saved PNG of isolated cluster {int(cluster_id)} to {png_output_file}")

        tif_output_file = f'{mainDir}/Isolated_Cluster_{int(cluster_id)}.tif'
        with rasterio.open(tif_output_file, 'w', **meta) as dst:
            dst.write(highlighted_image.astype(rasterio.uint8), 1)
        print(f"Saved georeferenced TIFF of isolated cluster {int(cluster_id)} to {tif_output_file}")

def plot_pca_scatter_clusters(valid_image_2d, cluster_indices, output_dir, sample_fraction=0.05):
    """
    Plots PCA-reduced 2D scatterplot of clustered data points, colored by cluster.
    """
    print("Running PCA scatter plot...")

    # Downsample if needed
    if sample_fraction < 1.0:
        np.random.seed(42)
        sample_size = int(len(valid_image_2d) * sample_fraction)
        indices = np.random.choice(len(valid_image_2d), sample_size, replace=False)
        data_sample = valid_image_2d[indices]
        cluster_sample = cluster_indices[indices]
    else:
        data_sample = valid_image_2d
        cluster_sample = cluster_indices

    # PCA
    pca = PCA(n_components=2, random_state=1)
    scaler = StandardScaler()
    data_sample_scaled = scaler.fit_transform(data_sample)
    data_pca = pca.fit_transform(data_sample_scaled)

    palette = sns.color_palette("husl", len(np.unique(cluster_sample)))

    # Plot
    df = {
        'PC1': data_pca[:, 0],
        'PC2': data_pca[:, 1],
        'Cluster': cluster_sample
    }

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=df['PC1'],
        y=df['PC2'],
        hue=df['Cluster'],
        palette=palette,
        s=8,
        alpha=0.6,
        linewidth=0,
        legend='full'
    )

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Projection of Clusters')
    plt.grid(True)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
    plt.tight_layout()

    # Save
    plot_path = os.path.join(output_dir, 'PCA_Scatter_Clusters.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f" PCA scatter plot saved to: {plot_path}")

#STATS :)
def log_cluster_statistics(valid_image_2d, cluster_indices, output_dir, feature_names=None,
                           low_content_threshold=0.01, high_variance_factor=3):
    """
    Logs statistics for each cluster and flags low-content or high-variance clusters.
    """
    print("Logging cluster statistics...")
    unique_clusters = np.unique(cluster_indices)
    num_features = valid_image_2d.shape[1]
    total_pixels = len(cluster_indices)

    if feature_names is None:
        feature_names = [f"Band {i+1}" for i in range(num_features)]

    global_std = np.nanstd(valid_image_2d, axis=0)

    lines = []
    stats_data = []

    for cluster in unique_clusters:
        mask = cluster_indices == cluster
        cluster_data = valid_image_2d[mask]
        pixel_count = cluster_data.shape[0]
        pixel_percentage = pixel_count / total_pixels * 100
        mean_vals = np.nanmean(cluster_data, axis=0)
        std_vals = np.nanstd(cluster_data, axis=0)

        # Flags
        low_content = pixel_percentage < (low_content_threshold * 100)
        high_variance = np.any(std_vals > high_variance_factor * global_std)

        lines.append(f"\n=== Cluster {int(cluster)} ===")
        lines.append(f"Pixel count: {pixel_count} ({pixel_percentage:.2f}%)")
        lines.append(f"Low content flag: {'YES' if low_content else 'no'}")
        lines.append(f"High variance flag: {'YES' if high_variance else 'no'}")
        lines.append("Feature Means:")
        for name, mean in zip(feature_names, mean_vals):
            lines.append(f"  {name}: {mean:.4f}")
        lines.append("Feature Standard Deviations:")
        for name, std in zip(feature_names, std_vals):
            lines.append(f"  {name}: {std:.4f}")

        stats_data.append({
            "Cluster": int(cluster),
            "Pixel Count": pixel_count,
            "% of Total Pixels": pixel_percentage,
            "Low Content": low_content,
            "High Variance": high_variance,
            **{f"Mean_{name}": mean for name, mean in zip(feature_names, mean_vals)},
            **{f"Std_{name}": std for name, std in zip(feature_names, std_vals)},
        })

    # Write text report
    report_path = os.path.join(output_dir, "Cluster_Stats_Report.txt")
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"Cluster statistics report saved to: {report_path}")

    # Write CSV report
    csv_path = os.path.join(output_dir, "Cluster_Stats_Report.csv")
    df_stats = pd.DataFrame(stats_data)
    df_stats.to_csv(csv_path, index=False)
    print(f"Cluster statistics CSV saved to: {csv_path}")

# MAIN
def main():
    mainDir = 'Insert your model.pkl file path here'
    outputDir = 'Insert your output folder filepath'

    model_path = os.path.join(mainDir, 'gmm_model_morphologic.pkl')
    new_image_path = '~Test/Test_morphologic_cube.tif' # Insert here the path to the Test raster. e.g., Test_morphologic_cube.tif which contains a higher-res. DTM
    os.makedirs(outputDir, exist_ok=True)

    # Load the saved GMM model
    gmm = joblib.load(model_path)
    print(f"Loaded GMM model from {model_path}")

    # Preprocess the new image
    new_image_vector, new_image_shape, new_image_meta = preprocess_image(new_image_path)

    nodata_value = -999
    valid_mask = ~np.any(new_image_vector == nodata_value, axis=1)
    new_image_vector_valid = new_image_vector[valid_mask]

    predicted_clusters_valid = gmm.predict(new_image_vector_valid)
    predicted_clusters = np.full((new_image_vector.shape[0],), -999, dtype='int16')
    predicted_clusters[valid_mask] = predicted_clusters_valid

    print("Predicted size:", predicted_clusters.shape)
    print("Expected shape:", new_image_shape)
    print("Trying to reshape into:", new_image_shape[1], new_image_shape[2])
    print("Expected total:", new_image_shape[1] * new_image_shape[2])

    # reshape
    expected_size = new_image_shape[1] * new_image_shape[2]
    actual_size = predicted_clusters.size  # Number of predictions from GMM

    if actual_size == expected_size:
        predicted_cluster_matrix = predicted_clusters.reshape(new_image_shape[1], new_image_shape[2])
    else:
        raise ValueError(f"Mismatch: Predicted clusters ({actual_size}) "
                         f"cannot be reshaped into ({new_image_shape[1]}, {new_image_shape[2]}). "
                         f"Check input dimensions and feature consistency.")

    num_features = new_image_vector.shape[1]
    feature_names = [f'Band {i + 1}' for i in range(num_features)]

    # PLOTS AND SAVINGS
    save_predicted_cluster_matrix(
        predicted_cluster_matrix=predicted_cluster_matrix,
        output_dir=outputDir,
        original_meta=new_image_meta
    )

    plot_individual_clusters(new_image_path, predicted_cluster_matrix, outputDir)

    plot_pca_scatter_clusters(
        valid_image_2d=new_image_vector_valid,
        cluster_indices=predicted_clusters_valid,
        output_dir=outputDir,
        sample_fraction=0.05
    )

    plot_feature_distributions_by_cluster(
        valid_image_2d=new_image_vector_valid,
        cluster_indices=predicted_clusters_valid,
        output_dir=outputDir,
        feature_names=feature_names,
        sample_fraction=0.05
    )

    # STATISTICS
    log_cluster_statistics(
        valid_image_2d=new_image_vector_valid,
        cluster_indices=predicted_clusters_valid,
        output_dir=outputDir,
        feature_names=[f'Band {i + 1}' for i in range(new_image_vector.shape[1])]
    )

if __name__ == '__main__':
    main()
