# RNA dataset

**Time-course transcriptomics review (RNA):** This manually labeled dataset consists of PubMed titles obtained for a systematic review on time-course transcriptomics (training *N = 1656*; validation *N = 184*; testing *N = 460*). Titles are labeled according to relevance, with a positive label indicating that a given title is pertinent to the review and a negative label indicating irrelevance. Three labeling functions were written for this study.

**License:** These datasets are licensed under the Open Data Commons Open Database License (ODbL) v1.0 [(https://opendatacommons.org/licenses/odbl/1-0/)](https://opendatacommons.org/licenses/odbl/1-0/). Any use of these datasets must abide by the terms of this license.

- `timecourse_rna_titles_2300.csv`: This file contains the raw PubMed data.
- `rna_*.csv`: These files are the train-val-test splits of `timecourse_rna_titles_2300.csv`.
- `rna_lf_matrix_*.csv`: These files are the train-val-test splits of the labeling function output matrix.
  - Column `Label` represents groundtruth labels.
  - Column `Snorkel` represents labels assigned by Snorkel.
- `rna_lfs.py`: Labeling functions written for this dataset.
