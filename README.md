# u_Map
u_Map contains a clustering tool based on Gaussian Mixture Models (GMMs) designed to generate classification maps of Mercury’s surface using topographic and spectral information derived from MESSENGER datasets.

The workflow is organized into sequential scripts, numbered according to the order in which they should be executed. Each step produces outputs required by the following stage.

⸻
RESPOSITORY STRUCTURE

1_Fitting_README.txt    
2_Fitting.py                   

3_Evaluation_README.txt  
4_Evaluation.py          

5_Predictions_README.txt 
6_Predictions.py          

BIC.py                    
boxplots.py               
⸻
WORKFLOW

The numbered scripts must be run in sequence:

Step 1–2: Model fitting
	•	2_Fitting.py performs Gaussian Mixture Model fitting on the selected input cube.
	•	The fitted model is saved and later reused for evaluation and prediction.

Instructions, required inputs, and outputs are described in: 1_Fitting_README.txt


Step 3–4: Model evaluation
	•	4_Evaluation.py evaluates clustering performance against ground-truth information.
	•	This stage uses outputs generated during fitting.

Instructions are provided in:3_Evaluation_README.txt


Step 5–6: Prediction on new raster data
	•	6_Predictions.py applies the trained model to new multiband rasters.
	•	It generates predicted cluster maps, isolated clusters, and statistical summaries.

Instructions are provided in: 5_Predictions_README.txt

⸻
ADDITIONAL SCRIPTS

BIC.py

This script can be used to estimate the optimal number of clusters through Bayesian Information Criterion (BIC) analysis before model fitting.
It is optional, but recommended when the number of clusters is not predefined.

⸻
boxplots.py

This script generates the boxplots used in the manuscript for visual comparison of feature distributions among clusters.
It is optional and intended for post-processing and figure generation.


⸻
INPUT CUBES

Input cubes, ground truth matrices, and test cubes are available for download at: https://figshare.com/s/2a7c78ac2964baf99267

The raster cubes provided for manuscript reproduction:
	•	are already cropped
	•	are already projected
	•	have the same spatial resolution
	•	are already balanced

This means they are ready to be directly loaded into the fitting workflow without additional preprocessing.

⸻
USAGE NOTES

File paths and file names must be adapted before each run.

For every script:
	•	input directories must be updated
	•	output directories must be updated
	•	raster names and model names may need adjustment depending on the selected cube

Several variables are explicitly defined within each script for this purpose.

The code is fully commented, and function names are intentionally explicit to make the workflow easier to understand and adapt 

⸻
COMPUTATIONAL NOTES

Several scripts can become memory intensive, especially for large multiband rasters.

For large datasets, execution on a remote server or a high-RAM workstation are strongly suggested.
