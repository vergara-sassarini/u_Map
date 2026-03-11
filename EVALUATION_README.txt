OVERVIEW

This script evaluates a predicted classification raster against a ground-truth (GT) raster and generates quantitative metrics together with graphical outputs.

It is designed for classification products where predicted cluster labels are manually reassigned to reference classes through a user-defined mapping dictionary. In this workflow, predicted labels are matched to GT classes, allowing direct comparison between clustering results and reference geological units.

The script computes overall and per-class performance metrics and produces spatial and statistical plots to support interpretation of the classification results.


⸻

INPUTS

The user must define:
	•	pred_path: path to the predicted raster
	•	gt_path: path to the ground-truth raster
	•	output_path: folder where outputs will be saved
	•	nodata_value: nodata value used in both rasters
	•	mapping: dictionary linking predicted labels to GT classes

⸻

METRICS COMPUTED

The evaluation includes:
	•	Overall Accuracy
	•	Overall Purity
	•	Per-class Accuracy
	•	F2-score
	•	Precision / Recall
	•	Adjusted Rand Index (ARI)
	•	Normalized Mutual Information (NMI)

⸻
OUTPUTS

spectral_ground_truth.png --> Ground-truth raster classified with target classes
spectral_predictions.png --> Raw predicted cluster raster before reassignment
spectral_reassigned_predictions.png --> Predicted raster after mapping clusters to GT classes
spectral_plots.png --> Composite figure showing GT, predictions, and reassigned predictions side by side
confusion_matrix_selected_classes.png --> Percentage confusion matrix for evaluated classes
error_map.png --> Spatial map of correct vs incorrect predictions
class_accuracy_maps.png --> Per-class spatial accuracy maps
class_distribution.png --> GT vs predicted pixel counts per class
precision_recall.png --> Per-class precision and recall comparison
per_class_spicy_metrics.png --> Accuracy, purity, and F2-score per class
evaluation_metrics_selected_classes.txt --> Full text report of global and per-class metrics
metrics_summary.csv --> Tabulated summary of per-class metrics


⸻

NOTES

The mapping dictionary allows multiple predicted clusters to be assigned to the same GT class. The script automatically checks for:
	•	duplicated GT assignments
	•	unmapped GT classes
	•	NaN consistency between rasters

Warnings are printed in the terminal when inconsistencies are detected.
