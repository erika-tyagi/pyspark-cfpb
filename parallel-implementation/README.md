### Parallel Implementation  
This folder contains the PySpark implementation of my pipeline. The notebooks are intended to be run on an EMR cluster – although setup intructions for running both on a cluster and in a Google Colab notebook are provided in the files. 

- `01_download-and-store.py` loads the raw data from the CFPB website and stores it in an S3 bucket 
- `02_explore-data.ipynb` includes exploratory analysis 
- `03_build-models.ipynb` contains the full machine learning pipeline 