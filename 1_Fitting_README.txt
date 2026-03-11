OVERVIEW

This script performs Gaussian Mixture Model (GMM) clustering on a multiband raster dataset and generates clustering outputs, statistics, and diagnostic plots.

It is designed for spectral, morphological, or morpho-spectral cubes where each valid raster pixel is treated as a multidimensional feature vector.


⸻

INSTRUCTIONS

Before running the script:
	1)	Set the input raster path in:

	>input_directory = '...'
	>filepath = os.path.join(input_directory, 'your_input_file.tif')


	2)	Set the output directory in:
	>output_directory = '...'


	3)	Set the desired number of clusters *manually:
	>num_clusters = 11

* If the number of clusters is unknown, use the BIC script to determine the optimal number using a local or the global minima of the bic/clusters curve.


	4)	Keep the covariance options list as follows to test the 4 possibilities:
	>covariance_options = ['full', 'diag', 'tied', 'spherical'] 


	5)	The holdout fraction is controlled here:
	>holdout_percentage = 0.2

Meaning: 80% TRAINING PIXELS | 20% HOLDOUT PIXELS 


⸻
WORKFLOW

The script performs the following steps:
	1.	Load multiband raster
	2.	Remove NoData pixels (here set as = -999)
	3.	Split valid pixels into training (80%) and holdout (20%)
	4.	Fit GMM models using different covariance types
	5.	Select best covariance using BIC
	6.	Predict clusters for training and holdout pixels
	7.	Save raster outputs
	8.	Compute posterior probabilities
	9.	Compute feature statistics per cluster
	10.	Perform PCA (for 2D visualization purposes)
	11.	Generate diagnostic plots

⸻
 OUTPUTS

gmm_model_spectral.pkl --> Saved fitted GMM model for future prediction

Cluster_Matrix_spectral_training.tif --> Clustered raster for training pixels (80%)

Cluster_Matrix_spectral.tif --> Clustered raster for holdout pixels (20%)

holdout_binary_mask_spectral.tif --> Binary mask showing holdout pixels (1), training pixels (0), nodata (-999)

posterior_probability.txt --> Posterior probabilities for each training pixel, plus cluster-wise probability statistics

feature_stats_per_cluster.txt --> Mean and standard deviation of each feature for every cluster

gmm_clustering_log_spectral.txt --> Summary log of clustering results and pixel counts per cluster

gmm_computational_stats.txt --> Computational statistics including runtime, BIC, covariance type, and pixel counts

Cluster_Histogram.png --> Histogram of cluster pixel distribution

PCA_Explained_Variance_Pareto_Chart.png --> PCA explained variance plot

Gaussian_Distributions_with_Actual_Data_PCA_Component_1.png --> Gaussian mixture distributions projected onto PCA component 1

Cluster_Scatter_PCA_PC1_vs_PC2.png --> PCA scatter plot of clustered training pixels


⸻
NOTES
	•	NoData pixels are excluded automatically before fitting.
	•	Cluster numbering starts at 0.
	•	BIC is used only for covariance selection, not for determining cluster number.
	•	PCA is computed only on training pixels.
