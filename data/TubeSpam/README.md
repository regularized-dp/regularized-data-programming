# TubeSpam dataset

The TubeSpam dataset  (Alberto et al 2015) (training *N = 1407*; validation *N = 157*; testing *N = 392*) is applied to 10 labeling functions defined for this dataset in the Snorkel documentation [(https://github.com/snorkel-team/snorkel-tutorials)](https://github.com/snorkel-team/snorkel-tutorials). Note that these datasets are derived from the original TubeSpam dataset (total *N = 1961*) rather than the truncated version used by Snorkel (total *N = 1836*).

>Alberto, T. C., Lochter, J. V. & Almeida, T. A. TubeSpam: Comment Spam Filtering on YouTube. in 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA) 138–143 (IEEE, 2015). doi:10.1109/ICMLA.2015.37.

- `TubeSpam.csv`: This file concatenates the CSV files contained in `TubeSpam_YouTube-Spam-Collection-v1`, obtained in March 2022 from [https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection](https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection).
- `TubeSpam_*.csv`: These files are the train-val-test splits of `TubeSpam.csv`.
- `TubeSpam_lf_matrix_*.csv`: These files are the train-val-test splits of the labeling function output matrix.
  - Column `Label` represents groundtruth labels.
  - Column `Snorkel` represents labels assigned by Snorkel.
