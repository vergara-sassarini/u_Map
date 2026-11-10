"""
u_Map - 4.1_Evaluation_Cross-tabs.py (added in v1.1.0)

Cross-tabulation of RAW predicted cluster labels against labeled data (LD) classes,
with no reassignment (remapping) of clusters to LD classes.

For each run the script writes:
  - a counts matrix (LD classes as rows, predicted clusters as columns);
  - a row-normalized matrix: distribution of each LD class across clusters;
  - a column-normalized matrix: LD composition of each cluster;
  - two heatmaps (row- and column-normalized, LD classes as rows);
  - Normalized Mutual Information (NMI, arithmetic-mean normalization, as in
    scikit-learn) and Adjusted Rand Index (ARI), computed from the counts matrix;
  - a JSON + TXT record of inputs, outputs, run parameters and (optionally) the fitted model.

The prediction raster and the LD raster must share the same grid (size, transform, CRS).
If they do not (e.g., a 222 m/pixel prediction against a 450 m/pixel LD), resample the
prediction onto the LD grid first with nearest-neighbour resampling, e.g.:
  gdalwarp -tr 450 450 -te <xmin ymin xmax ymax of the LD> -r near pred.tif pred_on_LD_grid.tif

See 3.1_Evaluation_Cross-tabs_README.txt for details.
For the alternative evaluation based on reassigning clusters to LD classes, see 4_Evaluation.py.
"""

import os
import sys
import json
import shutil
import hashlib
import platform
import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import sklearn
import joblib

# ============================================================================ USER SETTINGS
# Insert here the folder where outputs will be saved
OUTPUT_DIR = '~/u_Map_outputs/evaluation'

NODATA = -999          # nodata value used in the u_Map rasters
HASH_INPUTS = True     # SHA-256 of every input file in the run record; set False for very large rasters
SHOW_PLOTS = False     # set True to display the figures interactively

# Label dictionaries: raster value -> display name
CM_CLUSTERS = {i: f'C{i}' for i in range(8)}
CS_CLUSTERS = {i: f'C{i}' for i in range(11)}
CMS_CLUSTERS = {i: f'C{i}' for i in range(17)}
MORPHO_LD = {0: 'sp', 1: 'imp', 2: 'icp', 3: 'cfs', 4: 'cfh',
             5: r'C$_3$', 6: r'C$_2$', 7: r'C$_1$'}   # Wright et al. (2019); C3 = well-preserved, C1 = heavily degraded
# Spectral units of Zambon et al. (2022). Check the coding of your LD raster with np.unique:
# if it is coded 0-10 instead of 1-11, use {i: f'SU{i + 1}' for i in range(11)}
SPECTRAL_LD = {i: f'SU{i}' for i in range(1, 12)}

# One entry per cross-tabulation. Comment out the runs you do not need.
#   pred      : predicted cluster raster (held-out rasters are written by 2_Fitting.py)
#   ld        : labeled-data raster on the same grid
#   model     : optional path to the fitted GMM (.pkl), recorded in the run metadata
RUNS = [
    dict(name='cm_holdout',
         title='CM clusters vs. morphologic LD (20% held-out set)',
         pred='~/Fitting_outputs/Cluster_Matrix_morphologic_holdout.tif',
         ld='~/Data/GT_morphologic.tif',
         pred_names=CM_CLUSTERS, ld_names=MORPHO_LD, model=None),
    dict(name='cs_holdout',
         title='CS clusters vs. spectral LD (20% held-out set)',
         pred='~/Fitting_outputs/Cluster_Matrix_spectral_holdout.tif',
         ld='~/Data/GT_spectral.tif',
         pred_names=CS_CLUSTERS, ld_names=SPECTRAL_LD, model=None),
    dict(name='cms_vs_morpho',
         title='CMS clusters vs. morphologic LD (20% held-out set)',
         pred='~/Fitting_outputs/Cluster_Matrix_morphospectral_holdout.tif',
         ld='~/Data/GT_morphologic.tif',
         pred_names=CMS_CLUSTERS, ld_names=MORPHO_LD, model=None),
    dict(name='cms_vs_spectral',
         title='CMS clusters vs. spectral LD (20% held-out set)',
         pred='~/Fitting_outputs/Cluster_Matrix_morphospectral_holdout.tif',
         ld='~/Data/GT_spectral.tif',
         pred_names=CMS_CLUSTERS, ld_names=SPECTRAL_LD, model=None),
    dict(name='cm_predicted_100pct',
         title='CM predicted clusters (100% of pixels) vs. morphologic LD',
         pred='~/Predictions/Morphologic_predicted_map_on_LD_grid.tif',   # resampled onto the LD grid
         ld='~/Data/GT_morphologic.tif',
         pred_names=CM_CLUSTERS, ld_names=MORPHO_LD, model=None),
]

# ============================================================================ PLOT STYLE
mpl.rcParams['font.family'] = ['PT Sans', 'DejaVu Sans']   # falls back if PT Sans is not installed
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12


# ============================================================================ PROVENANCE HELPERS
def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def file_info(path, do_hash=True):
    info = {'path': os.path.abspath(path), 'exists': os.path.exists(path)}
    if info['exists']:
        st = os.stat(path)
        info['size_bytes'] = st.st_size
        info['modified'] = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')
        if do_hash:
            info['sha256'] = sha256_of(path)
    return info


def raster_info(path, do_hash=True):
    info = file_info(path, do_hash)
    with rasterio.open(path) as src:
        info.update({
            'shape_rows_cols': [src.height, src.width],
            'bands': src.count,
            'dtype': str(src.dtypes[0]),
            'crs': src.crs.to_wkt() if src.crs else None,
            'pixel_size': [src.res[0], src.res[1]],
            'bounds': list(src.bounds),
            'declared_nodata': src.nodata,
        })
    return info


def model_info(model_path):
    info = file_info(model_path)
    m = joblib.load(model_path)
    info['class'] = type(m).__name__
    if hasattr(m, 'get_params'):
        info['hyperparameters'] = m.get_params()
    for attr in ('converged_', 'n_iter_', 'lower_bound_', 'n_features_in_',
                 'weights_', 'means_', 'covariances_'):
        if hasattr(m, attr):
            info[attr] = getattr(m, attr)
    return info


def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_dict(f, d, indent=4, skip=('crs',)):
    pad = ' ' * indent
    for k, v in d.items():
        if k in skip:
            f.write(f"{pad}{k}: (see JSON)\n")
        elif isinstance(v, dict):
            f.write(f"{pad}{k}:\n")
            _write_dict(f, v, indent + 4, skip)
        elif isinstance(v, (list, np.ndarray)) and np.size(v) > 12:
            f.write(f"{pad}{k}: array of shape {np.shape(v)} (see JSON)\n")
        else:
            f.write(f"{pad}{k}: {v}\n")


def save_run_metadata(output_dir, filename, inputs, outputs, run_params,
                      model_path=None, do_hash=True):
    try:
        script = os.path.abspath(__file__)
        shutil.copy(script, os.path.join(output_dir, f'{filename}_script_copy.py'))
    except NameError:
        script = 'interactive session (no script file)'

    meta = {
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'environment': {
            'python': sys.version.split()[0], 'platform': platform.platform(),
            'numpy': np.__version__, 'pandas': pd.__version__,
            'rasterio': rasterio.__version__, 'GDAL (via rasterio)': rasterio.__gdal_version__,
            'scikit-learn': sklearn.__version__, 'seaborn': sns.__version__,
            'matplotlib': mpl.__version__, 'script': script,
        },
        'inputs': {name: raster_info(p, do_hash) for name, p in inputs.items()},
        'run_parameters': run_params,
        'model': model_info(model_path) if model_path else 'not provided',
        'outputs': {os.path.basename(p): file_info(p, do_hash=False) for p in outputs},
    }

    json_path = os.path.join(output_dir, f'{filename}_run_metadata.json')
    txt_path = os.path.join(output_dir, f'{filename}_run_metadata.txt')
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2, default=_json_default)
    with open(txt_path, 'w') as f:
        f.write(f"RUN METADATA: {filename}\nCreated: {meta['created']}\n")
        for section in ('environment', 'inputs', 'run_parameters', 'model', 'outputs'):
            f.write(f"\n{section.upper()}\n")
            val = meta[section]
            if isinstance(val, dict):
                _write_dict(f, val)
            else:
                f.write(f"    {val}\n")
    print(f"Run metadata saved: {json_path}")


# ============================================================================ CORE
def read_label_raster(path):
    """Return the first band and a validity mask (excludes NODATA, declared nodata and NaN)."""
    with rasterio.open(path) as src:
        data = src.read(1)
        declared = src.nodata
        grid = (src.height, src.width, src.transform, src.crs)
    valid = data != NODATA
    if declared is not None and not np.isnan(declared):
        valid &= data != declared
    if np.issubdtype(data.dtype, np.floating):
        valid &= np.isfinite(data)
    return data, valid, grid


def nmi_ari_from_counts(counts):
    """NMI (arithmetic-mean normalization) and ARI from a contingency table.
    Identical to sklearn's normalized_mutual_info_score and adjusted_rand_score
    applied to the pixel labels, without expanding the table into pixel arrays."""
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum()
    a, b = c.sum(axis=1), c.sum(axis=0)
    nz = c > 0
    mi = np.sum(c[nz] / n * np.log(c[nz] * n / np.outer(a, b)[nz]))

    def entropy(x):
        x = x[x > 0] / n
        return -np.sum(x * np.log(x))

    h_sum = entropy(a) + entropy(b)
    nmi = 0.0 if h_sum == 0 else mi / (h_sum / 2)

    def pairs(x):
        return x * (x - 1) / 2

    s_ij, s_a, s_b, total = pairs(c).sum(), pairs(a).sum(), pairs(b).sum(), pairs(n)
    expected = s_a * s_b / total
    max_index = (s_a + s_b) / 2
    ari = 1.0 if max_index == expected else (s_ij - expected) / (max_index - expected)
    return nmi, ari


def plot_matrix(df, cbar_label, title, subtitle, path):
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * df.shape[1] + 2), max(5, 0.45 * df.shape[0] + 2)))
    sns.heatmap(df, annot=True, fmt=".1f", cmap='Blues', vmin=0, vmax=100,
                annot_kws={'size': 9 if df.shape[1] > 12 else 11},
                cbar_kws={'label': cbar_label}, ax=ax)
    ax.set_title(f'{title}\n{subtitle}', fontweight='bold')
    ax.set_xlabel('Predicted cluster')
    ax.set_ylabel('Labeled data (LD) class')
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def build_cross_tab(name, title, pred, ld, pred_names, ld_names, output_dir, model=None, do_hash=True):
    pred_path, ld_path = os.path.expanduser(pred), os.path.expanduser(ld)
    pred_data, pred_valid, pred_grid = read_label_raster(pred_path)
    ld_data, ld_valid, ld_grid = read_label_raster(ld_path)

    if pred_grid[:2] != ld_grid[:2]:
        raise ValueError(f"[{name}] grid size differs: prediction {pred_grid[:2]} vs LD {ld_grid[:2]}. "
                         "Resample the prediction onto the LD grid (nearest neighbour) first.")
    if not pred_grid[2].almost_equals(ld_grid[2]):
        print(f"WARNING [{name}]: geotransforms differ; check that the two rasters are aligned.")

    valid = pred_valid & ld_valid
    p = pred_data[valid].astype(np.int64)
    g = ld_data[valid].astype(np.int64)

    pred_labels = np.array(sorted(pred_names))
    ld_labels = np.array(sorted(ld_names))
    keep = np.isin(p, pred_labels) & np.isin(g, ld_labels)
    pi = np.searchsorted(pred_labels, p[keep])
    gi = np.searchsorted(ld_labels, g[keep])
    counts = np.bincount(gi * len(pred_labels) + pi,
                         minlength=len(ld_labels) * len(pred_labels)).reshape(len(ld_labels), len(pred_labels))

    extra_pred = np.unique(p[~np.isin(p, pred_labels)])
    extra_ld = np.unique(g[~np.isin(g, ld_labels)])
    if extra_pred.size or extra_ld.size:
        print(f"WARNING [{name}]: {int((~keep).sum()):,} valid pixels have labels missing from the dictionaries "
              f"(prediction: {extra_pred.tolist()}, LD: {extra_ld.tolist()}). Check the label coding.")

    nmi, ari = nmi_ari_from_counts(counts)
    print(f"[{name}] NMI = {nmi:.4f}   ARI = {ari:.4f}   N = {int(counts.sum()):,}")

    rows = [ld_names[k] for k in ld_labels]
    cols = [pred_names[k] for k in pred_labels]
    df_counts = pd.DataFrame(counts, index=rows, columns=cols)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = counts.sum(axis=1, keepdims=True)
        col_sums = counts.sum(axis=0, keepdims=True)
        rownorm = np.where(row_sums > 0, counts / row_sums * 100, 0)
        colnorm = np.where(col_sums > 0, counts / col_sums * 100, 0)
    df_row = pd.DataFrame(rownorm, index=rows, columns=cols)
    df_col = pd.DataFrame(colnorm, index=rows, columns=cols)

    outputs = {
        'counts': os.path.join(output_dir, f'{name}_counts.csv'),
        'rownorm': os.path.join(output_dir, f'{name}_rownorm_pct.csv'),
        'colnorm': os.path.join(output_dir, f'{name}_colnorm_pct.csv'),
        'row_png': os.path.join(output_dir, f'{name}_rownorm.png'),
        'col_png': os.path.join(output_dir, f'{name}_colnorm.png'),
    }
    df_counts.to_csv(outputs['counts'])
    df_row.to_csv(outputs['rownorm'])
    df_col.to_csv(outputs['colnorm'])
    plot_matrix(df_row, 'Row-normalized (%)', title, 'Distribution of each LD class across clusters', outputs['row_png'])
    plot_matrix(df_col, 'Column-normalized (%)', title, 'LD composition of each cluster', outputs['col_png'])

    run_params = {
        'title': title,
        'layout': 'LD classes as rows, predicted clusters as columns; no reassignment of clusters to LD classes',
        'normalization': 'rownorm = distribution of each LD class across clusters; '
                         'colnorm = LD composition of each cluster',
        'metrics': 'NMI (arithmetic-mean normalization) and ARI, computed from the counts matrix',
        'nodata_value': NODATA,
        'predicted_label_names': {int(k): v for k, v in pred_names.items()},
        'ld_label_names': {int(k): v for k, v in ld_names.items()},
        'grid_shape': list(pred_data.shape),
        'pixels_valid_in_both': int(valid.sum()),
        'pixels_counted_in_matrix': int(counts.sum()),
        'pixels_valid_but_label_not_in_dictionaries': int(valid.sum() - counts.sum()),
        'cluster_pixel_totals': dict(zip(cols, counts.sum(axis=0).tolist())),
        'ld_class_pixel_totals': dict(zip(rows, counts.sum(axis=1).tolist())),
        'NMI': nmi, 'ARI': ari,
    }
    save_run_metadata(output_dir, name,
                      inputs={'prediction_raster': pred_path, 'ld_raster': ld_path},
                      outputs=list(outputs.values()), run_params=run_params,
                      model_path=os.path.expanduser(model) if model else None, do_hash=do_hash)
    return df_counts, nmi, ari


# ============================================================================ RUN
if __name__ == '__main__':
    out_dir = os.path.expanduser(OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for run in RUNS:
        _, nmi, ari = build_cross_tab(output_dir=out_dir, do_hash=HASH_INPUTS, **run)
        summary.append({'run': run['name'], 'NMI': round(nmi, 4), 'ARI': round(ari, 4)})
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, 'metrics_summary.csv'), index=False)
    print(pd.DataFrame(summary).to_string(index=False))
