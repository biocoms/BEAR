# BEAR - Bootstrap and Attribute Ranking

## Prerequisites

Before you begin, please download the following requirements:

- You have installed [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- You are using a machine running a compatible operating system (Windows, macOS, Linux).

## Setup
Follow these steps to set up your environment:
## Packages Used

This project uses the following packages:

- **pandas**: 1.5.3
- **numpy**: 1.23.5
- **scikit-learn**: 1.2.2
- **matplotlib**: 3.7.1
- **matplotlib-venn**: 0.11.6
- **skrebate**: 0.62
### 1. Create and activate the conda environment
```bash
conda create --name bear_final python=3.9
conda activate bear_final
```
### 2. Installing the packages
```bash
conda install pandas numpy scikit-learn matplotlib
pip install matplotlib-venn skrebate
```
### 3. Cloning the Repository
```bash
git clone https://github.com/biocoms/BEAR.git
cd BEAR
```
### 4. Running the pipeline
 ```bash
 sh run_pipeline.sh $input_file_path $number_of_features
```
### Example:
```bash
sh run_pipeline.sh inputs/test/multi_data.csv 20
```
## Submitting as a job file
```bash
qsub pipeline_trail.job
```
