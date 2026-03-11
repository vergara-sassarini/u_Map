OVERVIEW
This script applies a previously trained Gaussian Mixture Model (GMM) to a new multiband raster and generates cluster predictions together with statistical summaries and graphical outputs.

It is intended for prediction only: the model must already exist as a .pkl file produced during the fitting stage.

⸻
WORKFLOW

The script:
	•	loads a trained GMM model (.pkl)
	•	reads a multiband raster (.tif) --> use your test raster (e.g., the cube with the higher resolution DTM)
	•	reshapes raster bands into a pixel × feature matrix
	•	excludes NoData pixels
	•	predicts cluster membership for valid pixels
	•	reconstructs the cluster raster in georeferenced form
	•	exports visual and statistical diagnostics


⸻
INPUTS

The script requires:
	1)	Trained GMM model (generated as an output from "GMM_Fitting.py" script

	2)	Input multiband raster (e.g., Test_morphospectral_cube.tif)

⸻
OUTPUTS

Predicted_Cluster_Matrix.tif --> Georeferenced raster containing predicted cluster assignment for each valid pixel

Predicted_Cluster_Matrix.png --> Colored visualization of the predicted cluster raster

Isolated_Cluster_X.tif --> Georeferenced raster where only cluster X is highlighted and all other pixels are background

Isolated_Cluster_X.png --> Visual map showing the spatial distribution of cluster X only

PCA_Scatter_Clusters.png --> PCA projection of sampled pixels colored by predicted cluster membership

Feature_X_Band_Y_Violin.png --> Violin plot showing the distribution of each input feature across predicted clusters

Input_Feature_Correlation_Matrix.png --> Correlation matrix showing pairwise relationships between input features

Cluster_Stats_Report.txt --> Text report containing per-cluster pixel count, percentage, mean feature values, standard deviations, and warning flags

Cluster_Stats_Report.csv --> Tabular version of per-cluster statistics for further analysis

Feature_Violin_Plots/Feature_X_Band_Y_Violin.png --> Violin plot showing the distribution of each input feature across predicted clusters


⸻

Computational warning

This script is computationally demanding and can require large amounts of RAM, especially when:
	•	input rasters exceed several million pixels
	•	multiple feature plots are generated
	•	many clusters are present

Even in prediction mode, memory consumption can become substantial because the full raster is loaded into memory and reshaped before clustering.
