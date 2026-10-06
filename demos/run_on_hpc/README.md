# scGPT Slurm Job Submission on HPC

This repository contains the necessary scripts to run scGPT on a High Performance Computing (HPC) cluster using Slurm. The scripts are designed to be used with the [scGPT](https://github.com/bowang-lab/scGPT) repository.

## Usage

1. Clone this repository to your HPC cluster.
2. Update the `.env_example` file with your wandb API key and save it as `.env` in the same directory.
3. Create a `data` directory within the repository directory. Place the `.h5ad` file you want to use with scGPT into the `data` directory.
4. Adjust the `slurm_example.sh` script to match the correct paths and parameters for your HPC cluster. Ensure you update `$DATA_PATH` in the same script to point to your `.h5ad` file.

You can then use the following command to submit a job to the HPC cluster:
```bash
sbatch slurm.sh
```

The `slurm.sh` file will check whether the required software is installed and will proceed to set up the working directory if it is not. This includes: 

- Cloning the scGPT repository
- Downloading the necessary data
- Setting up the Python environment (using UV)
