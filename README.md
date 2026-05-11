# Environment Setup

Create and activate the conda environment:

```bash
conda create -n pipeline
conda activate pipeline
```

Install R and Python:

```bash
conda search r-base
conda install r-base=4.2.0
conda install python=3.8
```

Install required R packages:

```bash
conda install conda-forge::gmp
conda install conda-forge::r-seurat=4.4.0
conda install conda-forge::r-hdf5r
conda install bioconda::bioconductor-sc3
```

Install machine learning and CUDA dependencies:

```bash
conda install conda-forge::pot
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install additional R dependencies:

```bash
conda search -c conda-forge r-mclust
conda install -n pipeline -c conda-forge r-mclust=5.4.10
```

Install Python packages:

```bash
pip install scanpy
pip install anndata==0.8.0
pip install pandas==1.4.2
pip install scikit-learn==1.1.1
pip install scipy==1.8.1
pip install tqdm==4.64.0
pip install scikit-misc
```

Install R-Python interface:

```bash
conda install -c conda-forge rpy2=3.5.1
```

# Run the Project

After finishing the installation, you can run the main Jupyter notebooks in:

* `vae_ATACproject/`
* `vae_RNAproject/`
