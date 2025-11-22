# Create the environment

`conda create -n qcGEM python=3.8`

`conda activate qcGEM`

# Install package

- Deep learning model

`conda install pytorch==2.0.0 pytorch-cuda=11.7 pyg=2.5.2 pytorch-cluster pytorch-scatter pytorch-sparse cudatoolkit=11.7 -c pytorch -c nvidia -c pyg`

- tools

`conda install rdkit scikit-learn scipy sympy numpy pandas tqdm h5py`

- plot

`conda install seaborn matplotlib`

# Check package

`conda list`

---

> The PyG package suite is often difficult to install. Please ensure you configure the installation based on your specific machine specifications.
