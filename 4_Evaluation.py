import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import rasterio
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             fbeta_score, adjusted_rand_score, normalized_mutual_info_score,
                             precision_recall_fscore_support)
import requests
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PLEASE READ THE "EVALUATION_README.txt" FILE PROVIDED
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#Here, you can select the font type, size and color for your plots.
mpl.rcParams['font.family'] = 'PT Sans'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['legend.fontsize'] = 7
mpl.rcParams['figure.titlesize'] = 10

# File paths:
# pred_path = please insert the Predictions path in which your predicted matrix is saved. e.g., Cluster_matrix_morphologic_holdout.tif
# gt_path = please insert the ground truth (GT) matrix that matches your predictions file. e.g, GT_morphologic.tif
# output_path = please insert an output path to save the generated plots

pred_path = 'Insert your Input file path/Cluster_matrix_morphologic_holdout.tif'
gt_path = 'Insert your GT file path/GT_morphologic.tif'
output_path = 'Insert your output folder path'

nodata_value = -999 #please, select your Nan value!

# Load rasters
with rasterio.open(gt_path) as src_gt:
    gt_data = src_gt.read(1)
    gt_data = np.where(gt_data == nodata_value, np.nan, gt_data)

with rasterio.open(pred_path) as src_pred:
    pred_data = src_pred.read(1)
    pred_data = np.where(pred_data == nodata_value, np.nan, pred_data)

combined_nan = np.isnan(gt_data) | np.isnan(pred_data)
gt_data[combined_nan] = np.nan
pred_data[combined_nan] = np.nan

# This is a check for NaN consistency
nan_mismatch = np.sum(np.isnan(gt_data) != np.isnan(pred_data))
nan_mismatch_after = np.sum(np.isnan(gt_data) != np.isnan(pred_data))
print("NaN mismatch after cleaning:", nan_mismatch_after)
#if nan_mismatch > 0:
    print(f"Warning: {nan_mismatch} pixels have mismatched NaN values between GT and Prediction.")

# README: this dictionary contains the predicted labels that corresponds to the GT labels. e.g., mapping = {5:0} indicates that Predicted cluster 5 --> matches the GT class 0.
#This is the example for the morphologic cube
mapping = {5:0, 4:1, 6:1, 3:3, 7:3, 0:6}

# Detect multiple assignments to same GT class.
inverse_map = {}
for pred, gt in mapping.items():
    inverse_map.setdefault(gt, []).append(pred)
for gt_class, preds in inverse_map.items():
    if len(preds) > 1:
        print(f"Warning: GT class {gt_class} is assigned to multiple predicted classes: {preds}")

# Detect unmapped GT classes
gt_classes = set(np.unique(gt_data[~np.isnan(gt_data)]).astype(int))
mapped_gt_classes = set(mapping.values())
unassigned_gt = sorted(gt_classes - mapped_gt_classes)
if unassigned_gt:
    print(f"These GT classes have no predictions mapped to them: {unassigned_gt}")

label_colors = {
    0: '#FFB0B6', 1: '#FF6270', 2: '#8C4D3A', 3: '#FFFFA5',
    4: '#D1A35A', 5: '#FFFF62', 6: '#468639', 7: '#780000',
    8: '#FFA9FA', 9: '#C9CD00'#, 10: '#000089'
}

# Light the labels you need according to your input
# #class_names = {
#      0: 'SU1', 1: 'SU2', 2: 'SU3', 3: 'SU4',
#      4: 'SU5', 5: 'SU6', 6: 'SU7', 7: 'SU8',
#      8: 'SU9', 9: 'SU10', 10: 'SU11'
#  }

class_names = {
     0: 'Smooth Plains', 1: 'Intermediate Plains/Intercrater Plains', #2: 'Intercrater Plains',
     3: 'CFM Smooth', 4: 'CFM Hummocky', 5: 'CM well-preserved', 6: 'CM Degraded', 7: 'CM heavily-degraded',
     #8: 'SU9', 9: 'SU10', 10: 'SU11'
 }

# Reassign predictions
reassigned_pred = np.full_like(pred_data, np.nan)
for pred_label, gt_label in mapping.items():
    mask = np.isclose(pred_data, pred_label)
    reassigned_pred[mask] = gt_label
    print(f"Mapping pred={pred_label} → gt={gt_label}, matched pixels: {np.sum(mask)}")

# Legend labels
legend_labels = {i: f'SU{i+1}' for i in label_colors}

# Save function
def save_plot(fig, filename):
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


used_classes = sorted(np.unique(reassigned_pred[~np.isnan(reassigned_pred)]).astype(int))
used_colors = [label_colors[i] for i in used_classes]
used_cmap = ListedColormap(used_colors)
used_cmap.set_bad(color='white')


indexed_pred = np.full_like(reassigned_pred, np.nan)
for idx, cls in enumerate(used_classes):
    indexed_pred[reassigned_pred == cls] = idx

# Plot GT
fig, ax = plt.subplots(figsize=(10, 8))
masked_gt = np.ma.masked_invalid(gt_data)
cmap = ListedColormap([label_colors[i] for i in sorted(label_colors)])
cmap.set_bad(color='white')
ax.imshow(masked_gt, cmap=cmap, interpolation='none')
ax.set_title("Ground Truth", fontweight='bold')
ax.axis('off')
legend = [
    mpatches.Patch(color=color, label=class_names[i])
    for i, color in label_colors.items()
    if i in class_names
]
fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=6, frameon=False)
plt.show()
save_plot(fig, os.path.join(output_path, 'spectral_ground_truth.png'))


# Plot Predictions
fig, ax = plt.subplots(figsize=(10, 8))
masked_pred = np.ma.masked_invalid(pred_data)
tab20_raw = plt.get_cmap('tab20', int(np.nanmax(pred_data)) + 1) #for black nans
default_cmap = ListedColormap(tab20_raw.colors) #for black nans
default_cmap.set_bad(color='white') #for black nans
ax.imshow(masked_pred, cmap=default_cmap, interpolation='none')
ax.set_title("Predictions", fontweight='bold')
ax.axis('off')
legend = [mpatches.Patch(color=default_cmap(i), label=f'Class {i}') for i in np.unique(pred_data[~np.isnan(pred_data)]).astype(int)]
fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=6, frameon=False)
plt.show()
save_plot(fig, os.path.join(output_path, 'spectral_predictions.png'))

# Plot Reassigned Predictions
fig, ax = plt.subplots(figsize=(10, 8))
masked_indexed = np.ma.masked_invalid(indexed_pred)
ax.imshow(masked_indexed, cmap=used_cmap, interpolation='none')
ax.set_title("Reassigned Predictions", fontweight='bold')
ax.axis('off')
legend = [mpatches.Patch(color=label_colors[i], label=f'SU{i+1}') for i in used_classes]
fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=6, frameon=False)
plt.show()
save_plot(fig, os.path.join(output_path, 'spectral_reassigned_predictions.png'))


# Side-by-side composite plot
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
axes[0].imshow(masked_gt, cmap=cmap, interpolation='none')
axes[0].set_title('Ground Truth', fontweight='bold')
axes[0].axis('off')
axes[1].imshow(masked_pred, cmap=default_cmap, interpolation='none')
axes[1].set_title('Predictions', fontweight='bold')
axes[1].axis('off')
axes[2].imshow(masked_indexed, cmap=used_cmap, interpolation='none')
axes[2].set_title('Reassigned Predictions', fontweight='bold')
axes[2].axis('off')
fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=6, frameon=False)
plt.show()
save_plot(fig, os.path.join(output_path, 'spectral_plots.png'))


#EVALUATION: METRICS AND PLOTS

gt_flat = gt_data.flatten()
pred_flat = reassigned_pred.flatten()

# Mask for valid pixels
valid_mask = ~np.isnan(gt_flat) & ~np.isnan(pred_flat)
gt_valid = gt_flat[valid_mask].astype(int)
pred_valid = pred_flat[valid_mask].astype(int)

# Automatically derive classes present in reassigned_pred and GT
used_classes = sorted(np.unique(reassigned_pred[~np.isnan(reassigned_pred)]).astype(int))
class_names_ordered = [class_names[i] for i in used_classes]

# Filter for used classes
classes_mask = np.isin(gt_valid, used_classes) & np.isin(pred_valid, used_classes)
gt_selected = gt_valid[classes_mask]
pred_selected = pred_valid[classes_mask]

true_counts = np.array([np.sum(gt_selected == cls) for cls in used_classes])
pred_counts = np.array([np.sum(pred_selected == cls) for cls in used_classes])
print("True Counts per Class:", dict(zip(class_names_ordered, true_counts)))
print("Predicted Counts per Class:", dict(zip(class_names_ordered, pred_counts)))

# Remove classes with no support in either GT or Pred
non_zero_classes = [cls for cls, t, p in zip(used_classes, true_counts, pred_counts) if t > 0 and p > 0]
class_names_ordered = [class_names[i] for i in non_zero_classes]


if not non_zero_classes:
    print("No classes with support in both GT and Pred")
else:
    accuracy = accuracy_score(gt_selected, pred_selected)
    report = classification_report(gt_selected, pred_selected, labels=non_zero_classes, target_names=class_names_ordered, zero_division=0)
    conf_matrix = confusion_matrix(gt_selected, pred_selected, labels=non_zero_classes)
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    f2_scores = fbeta_score(gt_selected, pred_selected, labels=non_zero_classes, beta=2, average=None, zero_division=0)

    ari = adjusted_rand_score(gt_selected, pred_selected)
    nmi = normalized_mutual_info_score(gt_selected, pred_selected)

    purity_numerators = conf_matrix.max(axis=0)
    purity_denominators = conf_matrix.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        purity_per_class = purity_numerators / purity_denominators
        purity_per_class[~np.isfinite(purity_per_class)] = 0
    overall_purity = np.sum(purity_numerators) / np.sum(conf_matrix)

    metrics_text = f"Used Classes: {non_zero_classes}\n"
    metrics_text += f"Accuracy: {accuracy:.4f}\nOverall Purity: {overall_purity:.4f}\n"
    metrics_text += f"Adjusted Rand Index (ARI): {ari:.4f}\nNormalized Mutual Information (NMI): {nmi:.4f}\n"
    metrics_text += "\nPer-class Metrics:\n"
    for idx, purity, acc, f2 in zip(non_zero_classes, purity_per_class, per_class_accuracy, f2_scores):
        metrics_text += f"{class_names[idx]} (Class {idx}):\n"
        metrics_text += f"  Purity: {purity:.4f}\n  Accuracy: {acc:.4f}\n  F2-score: {f2:.4f}\n"
    metrics_text += "\nClassification Report:\n" + report

    with open(os.path.join(output_path, 'evaluation_metrics_selected_classes.txt'), 'w') as f:
        f.write(metrics_text)

    print("\nEvaluation Metrics:\n" + metrics_text)

    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_percent, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_names_ordered, yticklabels=class_names_ordered)
    plt.title('Confusion Matrix (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'confusion_matrix_selected_classes.png'), dpi=300)
    plt.show()

    precision, recall, fscore, support = precision_recall_fscore_support(gt_selected, pred_selected, labels=non_zero_classes, zero_division=0)
    df_metrics = pd.DataFrame({
        'Class': [class_names[i] for i in non_zero_classes],
        'Purity': purity_per_class,
        'Accuracy': per_class_accuracy,
        'F2-score': f2_scores,
        'Support': support
    })
    df_metrics.to_csv(os.path.join(output_path, 'metrics_summary.csv'), index=False)

    # Error map
    error_map = np.full(gt_data.shape, np.nan)
    error_map_flat = error_map.flatten()
    error_map_flat[valid_mask] = (gt_flat[valid_mask] != pred_flat[valid_mask])
    error_map = error_map_flat.reshape(gt_data.shape)

    # Make sure it's integers so colormap works as 0 → green, 1 → red
    error_map = error_map.astype('float')
    error_map = np.ma.masked_invalid(error_map)

    # Colormap
    error_cmap = ListedColormap(['green', 'red'])
    error_cmap.set_bad(color='white')

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(error_map, cmap=error_cmap, interpolation='none')
    plt.title("Error Map")
    plt.axis('off')
    plt.legend(handles=[
        mpatches.Patch(color='green', label='Correct'),
        mpatches.Patch(color='red', label='Incorrect')
    ], loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    plt.savefig(os.path.join(output_path, 'error_map.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Class Accuracy Maps
    fig, axes = plt.subplots(1, len(non_zero_classes), figsize=(5 * len(non_zero_classes), 5))
    for ax, cls in zip(axes, non_zero_classes):
        class_mask = (gt_data == cls)
        correct_predictions = (reassigned_pred == cls) & class_mask
        incorrect_predictions = (reassigned_pred != cls) & class_mask

        class_map = np.full(gt_data.shape, np.nan)
        class_map[correct_predictions] = 0  # index 0 → green
        class_map[incorrect_predictions] = 1  # index 1 → red

        masked_class_map = np.ma.masked_invalid(class_map)

        ax.imshow(masked_class_map, cmap=error_cmap, interpolation='none')
        ax.set_title(f"{class_names[cls]} (Class {cls})", fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'class_accuracy_maps.png'), dpi=300)
    plt.show()

    # Class Distribution
    gt_counts = [np.sum(gt_valid == cls) for cls in non_zero_classes]
    pred_counts = [np.sum(pred_valid == cls) for cls in non_zero_classes]
    x = np.arange(len(non_zero_classes))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.35/2, gt_counts, 0.35, label='GT', color='#a5c4b6')
    ax.bar(x + 0.35/2, pred_counts, 0.35, label='Pred', color='#f28c8c')
    ax.set_xticks(x)
    ax.set_xticklabels([class_names[i] for i in non_zero_classes], rotation=45)
    ax.legend()
    ax.set_title('Class Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'class_distribution.png'), dpi=300)
    plt.show()

    # Precision/Recall
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.35/2, precision, 0.35, label='Precision', color='#f4c27a')
    ax.bar(x + 0.35/2, recall, 0.35, label='Recall', color='#f2a177')
    ax.set_xticks(x)
    ax.set_xticklabels([class_names[i] for i in non_zero_classes], rotation=45)
    ax.legend()
    ax.set_title('Per-Class Precision and Recall')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'precision_recall.png'), dpi=300)
    plt.show()

    #FINAL COOL PLOT WITH METRICS
    df_sorted = df_metrics.sort_values(by='F2-score', ascending=False).reset_index(drop=True)
    cmap = mpl.colormaps['RdYlGn']  # Red = bad, Green = good

    norm = plt.Normalize(df_sorted['F2-score'].min(), df_sorted['F2-score'].max())
    colors = cmap(norm(df_sorted['F2-score']))

    x = np.arange(len(df_sorted))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width, df_sorted['Accuracy'] * 100, width, label='Accuracy', color='#6baed6')
    bars2 = ax.bar(x, df_sorted['Purity'] * 100, width, label='Purity', color='#74c476')
    bars3 = ax.bar(x + width, df_sorted['F2-score'] * 100, width, label='F2-score', color=colors)

    # Plot horizontal threshold line
    ax.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(len(x) - 0.5, 52, '50%', color='black', fontsize=10, va='bottom', ha='right')

    # Labels and aesthetics
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted['Class'], rotation=45, ha='right')
    ax.set_ylabel('Score (%)')
    ax.set_ylim(0, 110)
    ax.set_title('Per-Class Metrics (%)', fontweight='bold')
    ax.legend()
    plt.tight_layout()

    # Save & show
    plt.savefig(os.path.join(output_path, 'per_class_spicy_metrics.png'), dpi=300)
    plt.show()

























