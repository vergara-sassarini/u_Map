import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from sklearn.mixture import GaussianMixture
from time import perf_counter


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# USE THIS CODE SNIPPET FOR OPTIMAL NUMBER OF CLUSTERS SELECTION!
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def load_multiband_raster(filepath):
    with rasterio.open(filepath) as src:
        image = src.read()
        meta = src.meta.copy()
        no_data_value = src.nodata
    return image, meta, no_data_value


def preprocess_raster(image, no_data_value):
    num_bands, height, width = image.shape
    image_2d = image.reshape(num_bands, height * width).T

    if no_data_value is not None:
        image_2d[image_2d == no_data_value] = np.nan

    valid_mask = ~np.isnan(image_2d).any(axis=1)
    valid_image_2d = image_2d[valid_mask]
    return valid_image_2d, valid_mask, (height, width)

def fit_gmm_bic(data, k, covariance_type="full", seed=1, reg_covar=1e-5):
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        random_state=seed,
        reg_covar=reg_covar
    )
    gmm.fit(data)
    bic = gmm.bic(data)
    return bic


def plot_bic_curve(results_df, output_directory):
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["k"], results_df["bic"], marker="o", label="BIC")

    best_bic_k = results_df.loc[results_df["bic"].idxmin(), "k"]
    plt.axvline(best_bic_k, color="red", linestyle="--", label=f"BIC minimum (k={best_bic_k})")

    plt.title("BIC vs Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("BIC")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "BIC_vs_k.png"), dpi=300)
    plt.close()

    return best_bic_k


def run_bic_selection(
    data,
    output_directory,
    cluster_range=range(5, 35),
    covariance_type="full",
    seed=1,
    reg_covar=1e-5
):
    os.makedirs(output_directory, exist_ok=True)

    results = []
    start = perf_counter()

    print("Running GMM model selection using BIC...")
    print(f"Covariance type: {covariance_type}")
    print(f"Cluster range: {min(cluster_range)} to {max(cluster_range)}")

    for k in cluster_range:
        try:
            bic = fit_gmm_bic(
                data=data,
                k=k,
                covariance_type=covariance_type,
                seed=seed,
                reg_covar=reg_covar
            )
            results.append((k, bic))
            print(f"k={k:2d} | BIC={bic:.2f}")
        except Exception as e:
            print(f"k={k:2d} failed: {e}")

    if not results:
        raise RuntimeError("No GMM model converged for any k.")

    results_df = pd.DataFrame(results, columns=["k", "bic"])
    results_df.to_csv(os.path.join(output_directory, "GMM_BIC_Results.csv"), index=False)

    best_bic_k = plot_bic_curve(results_df, output_directory)

    elapsed = perf_counter() - start

    with open(os.path.join(output_directory, "BIC_summary.txt"), "w") as f:
        f.write("GMM model selection summary\n")
        f.write("---------------------------\n")
        f.write(f"Covariance type: {covariance_type}\n")
        f.write(f"Cluster range: {min(cluster_range)} to {max(cluster_range)}\n")
        f.write(f"Best BIC k: {best_bic_k}\n")
        f.write(f"Runtime: {elapsed:.2f} seconds\n")

    print("\nDone.")
    print(f"Best BIC k: {best_bic_k}")
    print(f"Results saved in: {output_directory}")

    return results_df, best_bic_k


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MAIN
# select your input (the folder that contains the cube you want to fit) and your output directories
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main():
    input_directory = "~/Inputs/"
    output_directory = "~/Outputs/"

    filepath = os.path.join(input_directory, "Fit_spectral_cube.tif")

    image, meta, no_data_value = load_multiband_raster(filepath)
    valid_image_2d, valid_mask, (height, width) = preprocess_raster(image, no_data_value)

    print(f"Loaded raster with shape: {image.shape}")
    print(f"Valid pixels: {len(valid_image_2d)}")

    # Optional downsampling for BIC selection
    np.random.seed(42)
    selection_fraction = 0.2
    sample_size = int(len(valid_image_2d) * selection_fraction)
    selection_indices = np.random.choice(len(valid_image_2d), sample_size, replace=False)
    downsampled_image_2d = valid_image_2d[selection_indices]

    print(f"Using {len(downsampled_image_2d)} pixels for BIC selection")

# !!SELECT THE cluster_range YOU PREFER! e.g., in this case the model will test from 5 to 35 clusters models.
    run_bic_selection(
        data=downsampled_image_2d,
        output_directory=output_directory,
        cluster_range=range(5, 35),
        covariance_type="full",
        seed=1,
        reg_covar=1e-5
    )


if __name__ == "__main__":
    main()
