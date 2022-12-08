# Regularized Data Programming with Bayesian Priors

Repository structure:

- `data`: Directory containing all data files used in experiments.
- `experiments_baselines`: Directory containing notebooks to reproduce Snorkel, CAGE, and supervised learning baselines.
- `experiments_low_data`: Directory containing notebooks to reproduce low-data regime experiments.
- `model_scripts`: Directory containing Python scripts implementing labeling models and notebooks that demo functionality.
  - `bayesian_dp.py`: Python script implementing regularized DP model.
  - `ratner_mle.py`: Python script implementing maximum likelihood DP model.
  - `demo_map_rna.ipynb`: Demonstrates MAP model functionality on RNA dataset.
  - `demo_map_tubespam.ipynb`: Demonstrates MAP model functionality on TubeSpam dataset.
- `requirements.txt`: All requirements for reproducibility.
