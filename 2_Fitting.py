import numpy as np
import rasterio
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import joblib
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import silhouette_samples, silhouette_score


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PLEASE READ THE "FITTING_README.txt" FILE PROVIDED
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# MAIN FUNCTIONS FOR FITTING
def load_multiband_raster(filepath):
    with rasterio.open(filepath) as src:
        image = src.read()  # Read all bands into a 3D array (shape: [bands, height, width])
        meta = src.meta.copy()  # Copy metadata to modify later
        no_data_value = src.nodata  # Get NoData value from the raster
    return image, meta, no_data_value

def preprocess_raster(image, no_data_value):
    num_bands, height, width = image.shape
    image_2d = image.reshape(num_bands, height * width).T

    if no_data_value is not None:
        image_2d[image_2d == no_data_value] = np.nan
    valid_mask = ~np.isnan(image_2d).any(axis=1)
    valid_image_2d = image_2d[valid_mask]  # Filter out invalid pixels
    return valid_image_2d, valid_mask, (height, width)

def save_cluster_raster(filepath, cluster_matrix, meta, height, width, no_data_value):
    meta.update(count=1, dtype=rasterio.int16, nodata=no_data_value)
    cluster_matrix = cluster_matrix.astype('int16')
    cluster_matrix = cluster_matrix.reshape(height, width)
    with rasterio.open(filepath, 'w', **meta) as dst:
        dst.write(cluster_matrix, 1)

# The binary mask will contain 20% of the pixels left out from training
def save_binary_mask(filepath, mask, meta, height, width):
    meta_mask = meta.copy()
    meta_mask.update(count=1, dtype=rasterio.int16, nodata=-999)

    mask = mask.reshape(height, width).astype('int16')
    with rasterio.open(filepath, 'w', **meta_mask) as dst:
        dst.write(mask, 1)

def fit_gmm_to_multidimensional_data(args):
    image_2d, num_clusters, covariance_type, seed, reg_covar = args
    gmm = GaussianMixture(n_components=num_clusters, covariance_type=covariance_type, random_state=seed, reg_covar=reg_covar)
    gmm.fit(image_2d)
    bic = gmm.bic(image_2d)
    return covariance_type, gmm, bic


# PLOT FUNCTIONS
def plot_cluster_histogram(cluster_indices, num_clusters, mainDir):
    plt.figure(figsize=(10, 6))
    plt.hist(cluster_indices, bins=np.arange(num_clusters + 1) - 0.5, rwidth=0.8, color='skyblue', alpha=0.7)
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Frequency")
    plt.xticks(np.arange(num_clusters))
    plt.grid(True)
    hist_output_path = os.path.join(mainDir, 'Cluster_Histogram.png')
    plt.savefig(hist_output_path, dpi=300)
    plt.close()
    print(f"Saved cluster histogram to {hist_output_path}")

# Function to help silhouette computation
def compute_silhouette_for_chunk(data_chunk, labels_chunk):
    return silhouette_samples(data_chunk, labels_chunk)

def plot_silhouette(valid_image_2d, cluster_indices, num_clusters, mainDir, sample_fraction=0.02):
    if len(valid_image_2d) > 1000:
        np.random.seed(1)
        sample_indices = np.random.choice(
            len(valid_image_2d),
            int(len(valid_image_2d) * sample_fraction),
            replace=False
        )
        valid_image_2d_sampled = valid_image_2d[sample_indices]
        cluster_indices_sampled = cluster_indices[sample_indices]
        print(f"Downsampling: Using {len(valid_image_2d_sampled)} points for silhouette calculation")
    else:
        valid_image_2d_sampled = valid_image_2d
        cluster_indices_sampled = cluster_indices

    silhouette_vals = silhouette_samples(valid_image_2d_sampled, cluster_indices_sampled)
    silhouette_avg = silhouette_score(valid_image_2d_sampled, cluster_indices_sampled)
    print(f"Average Silhouette Score: {silhouette_avg:.4f}")

    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(valid_image_2d_sampled) + (num_clusters + 1) * 10])

    y_lower = 10
    for i in range(num_clusters):
        ith_cluster_silhouette_values = silhouette_vals[cluster_indices_sampled == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.tab10(i / max(num_clusters, 1))
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title("Silhouette Plot for GMM Clustering")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([i * 0.2 for i in range(6)])

    plot_path = os.path.join(mainDir, 'Silhouette_Plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved silhouette plot to {plot_path}")

def plot_gaussian_bells_with_pca(gmm, pca, valid_image_pca, mainDir, component_index=0):
    num_clusters = gmm.n_components
    pca_values = valid_image_pca[:, component_index]

    plt.figure(figsize=(10, 6))
    sns.kdeplot(pca_values, color='black', linestyle='--', label='Actual Data', fill=True, alpha=0.2)

    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_

    gmm_pca_means = pca.transform(gmm.means_)
    gmm_pca_covariances = []
    if gmm.covariance_type == 'full':
        gmm_pca_covariances = [pca.components_ @ cov @ pca.components_.T for cov in gmm.covariances_]
    elif gmm.covariance_type == 'tied':
        tied_cov = pca.components_ @ gmm.covariances_ @ pca.components_.T
        gmm_pca_covariances = [tied_cov for _ in range(gmm.n_components)]
    elif gmm.covariance_type == 'diag':
        for diag in gmm.covariances_:
            diag_matrix = np.diag(diag)
            cov_pca = pca.components_ @ diag_matrix @ pca.components_.T
            gmm_pca_covariances.append(cov_pca)
    elif gmm.covariance_type == 'spherical':
        for s in gmm.covariances_:
            sph_matrix = np.eye(pca.n_components_) * s
            gmm_pca_covariances.append(sph_matrix)

    x_min, x_max = pca_values.min(), pca_values.max()
    x_vals = np.linspace(x_min, x_max, 1000)
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))

    for cluster in range(num_clusters):
        mean_pca = gmm_pca_means[cluster, component_index]
        #var_pca = gmm_pca_covariances[cluster][component_index, component_index]
        cov = gmm_pca_covariances[cluster]
        if cov.ndim == 1:
            var_pca = cov[component_index]
        else:
            var_pca = cov[component_index, component_index]
        std_dev_pca = np.sqrt(var_pca)
        y_vals = weights[cluster] * norm.pdf(x_vals, mean_pca, std_dev_pca)
        plt.plot(x_vals, y_vals, label=f'Cluster {cluster} (µ={mean_pca:.2f}, σ={std_dev_pca:.2f}, w={weights[cluster]:.2f})',
                 color=colors[cluster], alpha=0.6)
        plt.fill_between(x_vals, y_vals, alpha=0.3, color=colors[cluster])

    plt.title(f'Gaussian Mixtures vs. Actual Data Distribution for PCA Component {component_index + 1}')
    plt.xlabel(f'PCA Component {component_index + 1} Values')
    plt.ylabel('Density')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True)

    plot_path = os.path.join(mainDir, f'Gaussian_Distributions_with_Actual_Data_PCA_Component_{component_index + 1}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Gaussian distribution plot for PCA Component {component_index + 1} to {plot_path}")

def plot_scatter_pca_with_density(valid_image_pca, cluster_indices, mainDir, component_x=0, component_y=1):
    x = valid_image_pca[:, component_x]
    y = valid_image_pca[:, component_y]
    df = pd.DataFrame({
        f'PCA Component {component_x + 1}': x,
        f'PCA Component {component_y + 1}': y,
        'Cluster': cluster_indices
    })

    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x=f'PCA Component {component_x + 1}', y=f'PCA Component {component_y + 1}',
                    hue='Cluster', palette='tab10', alpha=0.6, edgecolor=None, s=10)
    plt.title(f'Cluster Scatter Plot with PCA Reduction: PC{component_x + 1} vs PC{component_y + 1}')
    plt.xlabel(f'PCA Component {component_x + 1}')
    plt.ylabel(f'PCA Component {component_y + 1}')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    plot_path = os.path.join(mainDir, f'Cluster_Scatter_PCA_PC{component_x + 1}_vs_PC{component_y + 1}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA-based cluster scatter plot to {plot_path}")

def plot_pareto_chart(pca, mainDir):
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    num_components = len(explained_variance)

    plt.figure(figsize=(12, 8))
    plt.bar(range(1, num_components + 1), explained_variance * 100, alpha=0.7, align='center',
            label='Individual Explained Variance', color='skyblue')
    plt.plot(range(1, num_components + 1), cumulative_variance * 100, color='red', marker='o',
             label='Cumulative Explained Variance')
    plt.xlabel('Principal Components')
    plt.ylabel('Variance Explained (%)')
    plt.title('Pareto Chart of PCA Explained Variance')
    plt.xticks(range(1, num_components + 1))
    plt.ylim(0, 110)
    plt.legend(loc='best')
    plt.grid(True, axis='y')
    plot_path = os.path.join(mainDir, 'PCA_Explained_Variance_Pareto_Chart.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Pareto chart of PCA explained variance to {plot_path}")


#MAIN FUNCTION
def main():
    overall_start_time = perf_counter()

    # Directories
    input_directory = '~/Inputs/'
    output_directory = '~/Outputs/'

    #+++++++++++++++++++++++++++
    # Filepaths:
    #PLEASE CHANGE THE FILENAMES WHEN WORKING WITH DIFFERENT CUBES: e.g., 'Fit_spectral_cube.tif' --> 'Fit_morphologic_cube.tif'
    # the lines below contain the filenames for the morphologic cueb as an example
    #++++++++++++++++++++++++++

    filepath = os.path.join(input_directory, 'Fit_morphologic_cube.tif')
    model_path = os.path.join(output_directory, 'gmm_model_morphologic.pkl')
    cluster_output_path_training = os.path.join(output_directory, 'Cluster_Matrix_morphologic_training.tif')
    cluster_output_path_holdout = os.path.join(output_directory, 'Cluster_Matrix_morphologic_holdout.tif')
    binary_mask_path = os.path.join(output_directory, 'holdout_binary_mask_morphologic.tif')
    log_file_path = os.path.join(output_directory, 'gmm_clustering_log_morphologic.txt')

    image, meta, no_data_value = load_multiband_raster(filepath)
    print(f"Loaded raster with shape {image.shape} (bands, height, width).")
    print(f"NoData value: {no_data_value}")

    # Preprocess raster
    valid_image_2d, valid_mask, (height, width) = preprocess_raster(image, no_data_value)
    print(f"Processed raster into a {valid_image_2d.shape} matrix (pixels x features).")

    total_valid_pixels = len(valid_image_2d)
    print(f"Total valid pixels (excluding NoData): {total_valid_pixels}")

    # This snippet is made to randomly select 80% of valid pixels for GMM fitting
    holdout_percentage = 0.2  # this equals to a training that uses 80% of the pixels, leaving 20% for test and evaluation
    num_holdout_samples = int(len(valid_image_2d) * holdout_percentage)
    np.random.seed(1)
    holdout_mask_indices = np.random.choice(len(valid_image_2d), num_holdout_samples, replace=False)
    holdout_image_2d = valid_image_2d[holdout_mask_indices]

    # The remaining 80% are the training data
    sampled_mask = np.ones(len(valid_image_2d), dtype=bool)
    sampled_mask[holdout_mask_indices] = False
    sampled_indices = np.where(sampled_mask)[0]
    sampled_image_2d = valid_image_2d[sampled_indices]

    # Create a binary mask with: holdout pixels (20%) = 1; valid training pixels (80%) = 0.
    binary_mask = np.full((height * width), -999, dtype=np.int16)
    temp_mask = np.zeros(len(valid_image_2d), dtype=np.int16)
    temp_mask[holdout_mask_indices] = 1
    binary_mask[valid_mask] = temp_mask
    save_binary_mask(binary_mask_path, binary_mask, meta, height, width)
    print(f"Binary mask saved to {binary_mask_path}")

    # Check prints to confirm coherence
    num_valid_pixels_in_binary_mask = np.sum(binary_mask == 1)
    print(f"Number of valid pixels in binary mask (should be 20% of valid pixels): {num_valid_pixels_in_binary_mask}")

    num_valid_pixels_in_sampled_image_2d = len(sampled_image_2d)
    print(f"Number of valid pixels in sampled_image_2d (should be 80% of valid pixels): {num_valid_pixels_in_sampled_image_2d}")

    total_valid_pixels = len(valid_image_2d)
    print(f"Total valid pixels (excluding NoData): {total_valid_pixels}")

    sum_of_pixels = num_valid_pixels_in_sampled_image_2d + num_valid_pixels_in_binary_mask
    print(f"Sum of training and holdout pixels: {sum_of_pixels}")
    if sum_of_pixels == total_valid_pixels:
        print("Numbers add up correctly.")
    else:
        print("Warning: Numbers do not add up correctly.")

    print(f"Training pixels count (80% of valid): {num_valid_pixels_in_sampled_image_2d}")
    print(f"Holdout pixels count (20% of valid): {num_valid_pixels_in_binary_mask}")



    #+++++++++++++++++++++++++++++++++++++
    # GAUSSIAN MIXTUR MODELS PARAMETERS:
    # The number of clusters for the morphospectral cube can be derived using the BIC approach.
    # The number of clusters for the morphological and spectral cube can be set by the user if a GT is known, or derived using BIC.
    #+++++++++++++++++++++++++++++++++++++

    num_clusters =8 # Number of clusters for the morphologic cube as an example
    covariance_options = ['full', 'diag', 'tied', 'spherical']
    seed = 1
    reg_covar = 1e-3

    print("Fitting GMM model...")
    gmm_start_time = perf_counter()

    args_list = [(sampled_image_2d, num_clusters, cov_type, seed, reg_covar) for cov_type in covariance_options]
    results = []

    for args in args_list:
        covariance_type, gmm, bic = fit_gmm_to_multidimensional_data(args)
        results.append((covariance_type, gmm, bic))
        print(f"Covariance type: {covariance_type}, BIC: {bic}")

    # BIC approach is used here to select the best covariance type for the selected number of clusters!
    best_bic = np.inf
    best_gmm = None
    best_covariance_type = None
    for covariance_type, gmm, bic in results:
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_covariance_type = covariance_type

    gmm_end_time = perf_counter()
    gmm_time = gmm_end_time - gmm_start_time
    print(f"Best covariance type: {best_covariance_type} with BIC: {best_bic}")
    print(f"GMM fitting completed in {gmm_time:.2f} seconds.")

    joblib.dump(best_gmm, model_path)
    print(f"Best GMM model saved to {model_path}")

    # Posterior probs
    print("Predicting clusters for training pixels...")
    sampled_cluster_indices = best_gmm.predict(sampled_image_2d)

    posterior_probs = best_gmm.predict_proba(sampled_image_2d)
    posterior_path = os.path.join(output_directory, 'posterior_probability.txt')
    np.savetxt(posterior_path, posterior_probs, fmt='%.6f', delimiter=',')

    with open(posterior_path, 'a') as f:
        f.write("\n\n---\nCluster-wise Posterior Probability Statistics\n")

        for cluster_id in range(num_clusters):
            cluster_mask = sampled_cluster_indices == cluster_id
            cluster_probs = posterior_probs[cluster_mask, cluster_id]  # Probabilities for own cluster only

            if len(cluster_probs) == 0:
                f.write(f"\nCluster {cluster_id}: No assigned pixels\n")
                continue

            f.write(f"\nCluster {cluster_id}:\n")
            f.write(f"  N assigned: {len(cluster_probs)}\n")
            f.write(f"  Max:    {np.max(cluster_probs):.4f}\n")
            f.write(f"  Min:    {np.min(cluster_probs):.4f}\n")
            f.write(f"  Mean:   {np.mean(cluster_probs):.4f}\n")
            f.write(f"  Median: {np.median(cluster_probs):.4f}\n")
            f.write(f"  Std:    {np.std(cluster_probs):.4f}\n")

    #FEATURE STATISTICS PER CLUSTER
    # Compute mean and std for each feature per cluster
    num_features = sampled_image_2d.shape[1]
    feature_stats_lines = []

    for cluster_id in range(num_clusters):
        cluster_data = sampled_image_2d[sampled_cluster_indices == cluster_id]
        feature_stats_lines.append(f"Cluster {cluster_id}")
        for feature_idx in range(num_features):
            feature_values = cluster_data[:, feature_idx]
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            feature_stats_lines.append(f"Feature {feature_idx + 1}: mean = {mean_val:.4f}, std = {std_val:.4f}")
        feature_stats_lines.append("")  # blank line between clusters

    # Save feature stats to a .txt file
    feature_stats_path = os.path.join(output_directory, 'feature_stats_per_cluster.txt')
    with open(feature_stats_path, 'w') as f:
        f.write('\n'.join(feature_stats_lines))

    print(f"Feature stats per cluster saved to {feature_stats_path}")

    print("Predicting clusters for holdout pixels...")
    holdout_cluster_indices = best_gmm.predict(holdout_image_2d)

    cluster_counts_training = np.bincount(sampled_cluster_indices, minlength=num_clusters)
    for i, count in enumerate(cluster_counts_training):
        print(f"Cluster {i}: {count} pixels in training data")

    cluster_counts_holdout = np.bincount(holdout_cluster_indices, minlength=num_clusters)
    for i, count in enumerate(cluster_counts_holdout):
        print(f"Cluster {i}: {count} pixels in holdout data")

    # Cluster_matrix_training is the matrix that will contain the output clustering = fitting results
    cluster_matrix_training = np.full((height * width), no_data_value, dtype='int16')
    temp_clusters_training = np.full(len(valid_image_2d), no_data_value, dtype='int16')
    temp_clusters_training[sampled_indices] = sampled_cluster_indices.astype('int16')
    cluster_matrix_training[valid_mask] = temp_clusters_training

    save_cluster_raster(cluster_output_path_training, cluster_matrix_training, meta, height, width, no_data_value)
    print(f"Cluster matrix for training data saved to {cluster_output_path_training}")

    #
    cluster_matrix_holdout = np.full((height * width), no_data_value, dtype='int16')
    temp_clusters_holdout = np.full(len(valid_image_2d), no_data_value, dtype='int16')
    temp_clusters_holdout[holdout_mask_indices] = holdout_cluster_indices.astype('int16')
    cluster_matrix_holdout[valid_mask] = temp_clusters_holdout

    # Save the cluster matrix for holdout data
    save_cluster_raster(cluster_output_path_holdout, cluster_matrix_holdout, meta, height, width, no_data_value)
    print(f"Cluster matrix for holdout data saved to {cluster_output_path_holdout}")

    # Perform PCA on the training data
    print("Performing PCA on the training data...")
    pca = PCA(n_components=0.95, random_state=1)
    sampled_image_pca = pca.fit_transform(sampled_image_2d)
    sampled_image_pca_2d = sampled_image_pca[:, :2]
    print(f"PCA completed. Explained variance ratios: {pca.explained_variance_ratio_}")

    # PLOTS WITH TRAINING DATA
    plot_pareto_chart(pca, output_directory)
    plot_gaussian_bells_with_pca(best_gmm, pca, sampled_image_pca, output_directory, component_index=0)
    plot_scatter_pca_with_density(sampled_image_pca, sampled_cluster_indices, output_directory, component_x=0,
                                  component_y=1)
    #plot_silhouette(sampled_image_2d, sampled_cluster_indices, num_clusters, output_directory)
    plot_cluster_histogram(sampled_cluster_indices, num_clusters, output_directory)

    # End the timer
    overall_end_time = perf_counter()
    total_runtime = overall_end_time - overall_start_time
    print(f"Total runtime: {total_runtime:.2f} seconds.")

    #COMPUTATIONAL PERFORMANCE STUFF FOR EACH STEP
    performance_stats_path = os.path.join(output_directory, 'gmm_computational_stats.txt')
    with open(performance_stats_path, 'w') as perf_file:
        perf_file.write("GMM Clustering Performance Statistics\n")
        perf_file.write("--------------------------------------\n")
        perf_file.write(f"Covariance type: {best_covariance_type}\n")
        perf_file.write(f"BIC: {best_bic:.2f}\n")
        perf_file.write(f"Number of clusters: {num_clusters}\n")
        perf_file.write(f"GMM fitting time: {gmm_time:.2f} seconds\n")
        perf_file.write(f"Total runtime: {total_runtime:.2f} seconds\n")
        perf_file.write(f"Training pixels: {num_valid_pixels_in_sampled_image_2d}\n")
        perf_file.write(f"Holdout pixels: {num_valid_pixels_in_binary_mask}\n")
        perf_file.write(f"Total valid pixels: {total_valid_pixels}\n")

    # Write all the information to a log file
    with open(log_file_path, 'w') as log_file:
        log_file.write("Gaussian Mixture Model Information:\n")
        log_file.write(f"Best covariance type: {best_covariance_type}\n")
        log_file.write(f"Best BIC: {best_bic}\n")
        log_file.write(f"GMM fitting time: {gmm_time:.2f} seconds\n\n")
        log_file.write("Clustering Results:\n")

        log_file.write("Training Data:\n")
        for i, count in enumerate(cluster_counts_training):
            log_file.write(f"Cluster {i}: {count} pixels\n")

        log_file.write("\nHoldout Data:\n")
        for i, count in enumerate(cluster_counts_holdout):
            log_file.write(f"Cluster {i}: {count} pixels\n")

        log_file.write(f"\nTotal runtime: {total_runtime:.2f} seconds\n")

    print(f"Log file saved to {log_file_path}")


if __name__ == "__main__":
    main()
