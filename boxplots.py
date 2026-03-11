import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# README: to use this boxplot script, you need to the "feature_summary_per_cluster.csv" generated during the Predictions step!!
csv_path = "Your .csv Input Directory"
df = pd.read_csv(csv_path)

required = {"cluster", "feature", "median", "q1", "q3", "iqr"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")

features = list(df["feature"].unique())

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
feat_color = {f: color_cycle[k % len(color_cycle)] for k, f in enumerate(features)}

# Plotting parameters
cluster_gap = 1.2   # spacing between clusters
box_width = 0.16    # width of each box
within_gap = 0.22   # spacing between feature boxes inside a cluster
feat_offsets = (np.arange(len(features)) - (len(features)-1)/2) * within_gap

# Saving directory (the same as the one containing the .csv file)
save_dir = Path(csv_path).parent

# Here, you can choose how many clusters you want to plot inside each panel, to improve visualization.
cluster_groups = [
    ("clusters_0_5", range(0, 6)),
    ("clusters_6_11", range(6, 12)),
    ("clusters_12_17", range(12, 18)),
]

for label, cl_range in cluster_groups:

    clusters = [c for c in sorted(df["cluster"].unique()) if c in cl_range]
    if len(clusters) == 0:
        print(f"[WARN] No clusters found for {label}. Skipping.")
        continue

    # boxplots parameters
    stats = []
    positions = []
    feat_for_box = []

    for i, c in enumerate(clusters):
        base_x = i * cluster_gap
        sub = df[df["cluster"] == c].set_index("feature")

        for j, f in enumerate(features):
            if f not in sub.index:
                continue

            med = float(sub.loc[f, "median"])
            q1  = float(sub.loc[f, "q1"])
            q3  = float(sub.loc[f, "q3"])
            iqr = float(sub.loc[f, "iqr"])

            whis_lo = q1 - 1.5 * iqr
            whis_hi = q3 + 1.5 * iqr

            stats.append({
                "med": med,
                "q1": q1,
                "q3": q3,
                "whislo": whis_lo,
                "whishi": whis_hi,
                "fliers": []
            })
            positions.append(base_x + feat_offsets[j])
            feat_for_box.append(f)

    fig, ax = plt.subplots(figsize=(14, 6))

    bp = ax.bxp(
        stats,
        positions=positions,
        widths=box_width,
        showfliers=False,
        patch_artist=True
    )

    # To color each box by feature
    for box, f in zip(bp["boxes"], feat_for_box):
        box.set_facecolor(feat_color[f])
        box.set_alpha(0.7)
        box.set_linewidth(1)

    #
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.8)

    cluster_centers = [i * cluster_gap for i in range(len(clusters))]
    ax.set_xticks(cluster_centers)
    ax.set_xticklabels([str(c) for c in clusters])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Feature value (z-score)")

    # Legend
    handles = [Patch(facecolor=feat_color[f], edgecolor="k", alpha=0.7, label=f) for f in features]
    ax.legend(handles=handles, title="Feature", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)

    plt.tight_layout()

    outpath = save_dir / f"boxplots_merged_{label}.png"  #choose the name according to the product you're saving
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {outpath}")
