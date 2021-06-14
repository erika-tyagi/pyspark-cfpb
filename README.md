For my project, I explore how large-scale computing techniques can make machine learning pipelines more efficient. Specifically, I build a pipeline to predict outcomes using the Consumer Financial Protection Bureau's (CFPB) [Consumer Complaints Database](https://www.consumerfinance.gov/data-research/consumer-complaints/). I primarily leverage PySpark on EMR clusters in my large-scale implementation, and I store my data in an S3 bucket to facilitate working within the AWS ecosystem. 

### Repository 
My serial implementation (using `pandas` and `scikit-learn`) of this pipeline is contained in the `serial-implementation` folder. My parallel implementation (using PySpark) is contained in the `parallel-implementation` folder).  

### Social Science Research Problem 
The CFPB publishes a database of over 1.6 million complaints filed since 2012 about consumer financial products and services that the agency sent to companies for a response. This database includes information about the company named, the relevant type of product and issue, a free form narrative description, and the company response to the complaint. I'm specifically trying to predict the company response, where this can fall into one of four categories: 
- Closed with explanation 
- Closed with monetary relief 
- Closed with non-monetary relief 
- Untimely response 

Predicting this outcome has a variety of potential applications at the intersection of social science and policy (e.g. anomaly detection, identifying complaints that deserve attention by the agency, allowing consumers to understand the likelihood of various outcomes when filing complaints, etc.). 

I use a [One-vs-Rest](https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest) (OvR) approach for this multi-class classification problem, which builds a single binary classifier for each of the four classes. I train the following types of classifiers: 
- [Logistic Regression](https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression) 
- [Decision Tree](https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier)
- [Support Vector Machine](https://spark.apache.org/docs/latest/ml-classification-regression.html#linear-support-vector-machine)
- [Gradient-Boosted Decision Tree](https://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-classifier) 
- [Naive Bayes](https://spark.apache.org/docs/latest/ml-classification-regression.html#naive-bayes)

I iterate over a range of hyperparameters for each of these classifiers using k-fold cross validation and F1-score as my primary evaluation metric to identify the best performing model. 

Three important aspects of this application make large-scale techniques particularly useful. 
- Multi-class classification is inherently computationally intensive. As noted, the OvR approach to multi-class classification requires building one model for each of the classes – which makes this particular problem four times more computationally expensive than binary classification. Of course, using a [One-vs-One](https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-one) approach (building a binary classifier for each pairwise combination of classes) would be even more expensive – particularly as the number of classes increases. 
- The dataset is highly class-imbalanced – over 80% of complaints fall into the Closed with explanation category. As a result, oversampling the minority classes makes the training dataset extremely large. Additionally, this oversampling process itself is computationally expensive. 
- Most of the features in the model are categorical and must be one-hot-encoded. Pre-processing while feature engineering can partly shrink the feature space (e.g. grouping companies with fewer than a certain number of complaints into an Other category, rather than one-hot-encoding all 5,000+ companies listed in the database) – but the feature space will inherently be sparse and high-dimensional with this type of data. As a result, large-scale techniques can be helpful in more efficiently storing this data and in training models. 

In short, predicting company responses to consumer complaints is a useful social science and policy problem itself. Moreover, the higher-level characteristics of this machine learning application (multi-class classification, class imbalance, and a sparse and high-dimensional feature space) make this a useful case study informing how other social science machine learning problems sharing these traits can benefit from large-scale computing techniques. 

### Large-Scale Approach
I first download the CFPB Consumer Complaints database directly from [the CFPB website in CSV format](http://files.consumerfinance.gov/ccdb/complaints.csv.zip), convert the dataset to Parquet format, and then store it in an S3 bucket. This streamlines accessing the data across my local machine and on EMR clusters, while also storing the data more efficiently. (See `01_download-and-store.py`). 

Next, I perform basic exploratory analysis in PySpark. Note that I'm using this dataset for another project for a course on deep learning and natural language processing, so I spent less time on this step than I would if I were unfamiliar with the data. (See `02_explore-data.ipynb`). 

I then use PySpark (leveraging the `pyspark.sql` and `pyspark.ml` libraries) to build the machine learning pipeline, which I run on an EMR cluster with the maximum allowed 8 m5.xlarge instances. Specifically, I perform the following steps (See `03_build-models.ipynb`): 
1. Load the data from [my S3 bucket]('s3://macs-30123-final-proj-tyagie/complaints.parquet') and perform basic pre-processing.
2. Engineer the relevant features described above. As noted, this requires indexing and one-hot-encoding the categorical features using PySpark's `StringIndexer` and `OneHotEncoderEstimator`. 
3. Split and resample the training data to account for class imbalance. I under-sample the majority class and over-sample the minority classes. Note that I provide code to do this two ways: (1) through bootstrapping, and (2) through a [synthetic over-sampling technique](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html) that requires converting PySpark's RDDs into either NumPy arrays or Pandas dataframes. 
4. Use k-fold cross-validation to select the best set of hyperparameters for each type of classifier with PySpark's `CrossValidator`, and then evaluate on the testing set. 

Given that this is a large-scale computing course (rather than a machine learning course), I'm less interested in building the best-performing classifier – and more interested in comparing the computation time between the PySpark implementation and the serial implementation (leveraging a standard `pandas` and `scikit-learn` workflow). This comparison based on training identical parameter grids is shown below: 

- Logistic Regression  
Serial: 0:34:30  |  Parallel: 0:05:19 
- Decision Tree   
Serial: 1:24:38  |  Parallel: 1:00:15
- Support Vector Machine  
Serial: 7:46:12  |  Parallel: 6:18:54 
- Gradient-Boosted Decision Trees  
Serial: 4:27:02  | Parallel:  N/A 
- Naive Bayes  
Serial: 0:02:03  | Parallel: 0:01:36  

Across the classifiers, the PySpark solutions were much faster – for example, up to 7 times as fast for the Logistic Regression. Moreover, if I were to cross-validate over a larger parameter grid or include more features, Spark's distributive framework and parallelism would likely make this gap even larger. Although my models themselves weren't incredibly strong (with micro- and macro-F1 scores below 0.7), leveraging PySpark makes further improving these models much more computationally feasible (e.g. through more sophisticated feature engineering, alternative resampling techniques, iterating over a wider range of hyperparamaters, etc.). 

Finally, although the PySpark implementation was much faster, it was generally much harder to work with in a multi-class classification setting. Specifically, there was less documentation and built-in functionality relative to other sections of the `mllib` library (and there are several open developer issues related to the functionality that does exist). Thus, while this large-scale approach has clear advantages in computational efficiency, there is also a tradeoff in built-in flexibility. 