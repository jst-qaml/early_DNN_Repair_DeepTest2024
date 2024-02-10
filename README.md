# More is Not Always Better: Exploring Early Repair of DNNs

This is the companion repository of the paper "More is Not Always Better: Exploring Early Repair of DNNs". This is a ICSE DeepTest workshop 2024 paper, that explores the best time to switch from training to repair through experiments.

## Setup

We have provided an environment.yml with all the packages required for running the experiments. Please install the environment and run it.
```
conda env create -f environment.yml
conda activate exp
```
## Datasets

In order to run the experiments, a download of the two datasets is necessary. We have provide the links to download here:

First, you need to create an `inputs` folder, where you will put the datasets. We recomend creating the folder inside the repository, but it can also be anywhere. If you choose to be outside the repository, some paths need to be changed in the experiment scripts.

- German Traffic Sign Recognition Benchmark (GTSRB):
    * Access [data-archive site](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html)
    * Download the following items:
        - `GTSRB_Final_Training_Images.zip`
        - `GTSRB_Final_Test_Images.zip`
        - `GTSRB_Final_Test_GT.zip`
    * Unzip the items and move them under `inputs` directory (e.g. `inputs/gtsrb/`)

- RNSA Penumonia Detection:
    * Access [Kaggle website](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
    * Download all the items from the Kaggle data section
    * If necessary, unzip the items and move them under `inputs` directory (e.g. `inputs/rsna_small/`)

GTSRB is a datatset containing 43 classes, while the RSNA Penumonia Detection dataset contains only 2. 

## How to use

A `run_search.py` script is provided in the root of the repository. The script can support multi-GPU processing and has the following specifications:
- Specify the number of GPUs on your machine and the initial GPU you want to use (in case you are working on a cluster)
    - Default parameters are set to 1 GPU, starting from 0
    - The script works on consecutive GPUs
    - You can also specify the minimum VRAM usage, for which a process stays in queue until the VRAM allocation of that GPU is less than specified.

- Additionally, you can specify the wait time for each indiviual step, for the processes to wait if the VRAM allocation is exceeded
- Specify the dataset and model that you want to use.
Additional models and datasets need to be specified inside the `settings` folder.
- Specify the class that you want to repair

## Data Visualization

After you run all the processes, you should have for each training run a repaired model inside `<model>_run<number>/negative/<repair_class>/` and its `result.txt`, which contains the performance metrics for the repair. Additionally, you can find a `test_class_metrics.txt` for each base training with the same metrics. 

An ipython notebook is provided in `deeptest_paper.ipynb` that contains all the plots related to the publication.

## Building blocks

In `cli.py` are the building blocks of the multi-GPU script available. You can use them to build your own personalized pipeline.


## Questions
If you have questions or any kind of problems, please open an issue or send me a message at andrei.mancu at tum.de .