# Scalable Multi-Agent Reinforcement Learning

## 1. Featured algorithms:

- Value Function Factorization with Variable Agent Sub-Teams (VAST) [1]

## 2. Implemented domains

All available domains are listed in the table below. The labels are used for the commands below (in 5. and 6.).

| Domain   		| Label            | Description                                                       |
|---------------|------------------|-------------------------------------------------------------------|
| Warehouse[4]  | `Warehouse-4`    | Warehouse domain with 4 agents in a 5x3 grid.					   |
| Warehouse[8]  | `Warehouse-8`    | Warehouse domain with 8 agents in a 5x5 grid. 					   |
| Warehouse[16] | `Warehouse-16`   | Warehouse domain with 16 agents in a 9x13 grid. 				   |
| Battle[20]    | `Battle-20`      | Battle domain with armies of 20 agents each in a 10x10 grid.      |
| Battle[40]    | `Battle-40`      | Battle domain with armies of 40 agents each in a 14x14 grid.      |
| Battle[80]    | `Battle-80`      | Battle domain with armies of 80 agents each in a 18x18 grid.      |
| GaussianSqueeze[200]    | `GaussianSqueeze-200` | Gaussian squeeze domain 200 agents.                |
| GaussianSqueeze[400]    | `GaussianSqueeze-400` | Gaussian squeeze domain 400 agents.                |
| GaussianSqueeze[800]    | `GaussianSqueeze-800` | Gaussian squeeze domain 800 agents.                |

## 3. Implemented MARL algorithms

The reported MARL algorithms are listed in the tables below. The labels are used for the commands below (in 5. and 6.).

| Baseline        | Label                  |
|-----------------|------------------------|
| IL              | `IL`                   |
| QMIX            | `QMIX`                 |
| QTRAN           | `QTRAN`                |

| VAST(VFF operator) | Label                  |
|--------------------|------------------------|
| VAST(IL)           | `VAST-IL`              |
| VAST(VDN)          | `VAST-VDN`         	  |
| VAST(QMIX)         | `VAST-QMIX`            |
| VAST(QTRAN)        | `VAST-QTRAN`     	  |

| VAST(assignment strategy) | Label                     |
|---------------------------|---------------------------|
| VAST(Random)              | `VAST-QTRAN-RANDOM`       |
| VAST(Fixed)               | `VAST-QTRAN-FIXED`        |
| VAST(Spatial)             | `VAST-QTRAN-SPATIAL`      |
| VAST(MetaGrad)            | `VAST-QTRAN`              |

## 4. Experiment parameters

The experiment parameters like the learning rate for training (`params["learning_rate"]`) or the number of episodes per epoch (`params["episodes_per_epoch"]`) are specified in `settings.py`. All other hyperparameters are set in the corresponding python modules in the package `vast/controllers`, where all final values as listed in the technical appendix are specified as default value.

All hyperparameters can be adjusted by setting their values via the `params` dictionary in `settings.py`.

## 5. Training

To train a MARL algorithm `M` (see tables in 3.) in domain `D` (see table in 2.) with compactness factor `eta`, run the following command:

    python train.py M D eta

This command will create a folder with the name pattern `output/N-agents_domain-D_subteams-S_M_datetime` which contains the trained models (depending on the MARL algorithm).

`train.sh` is an example script for running all settings as specified in the paper.

## 6. Plotting

To generate plots for a particular domain `D` and evaluation mode `E` as presented in the paper, run the following command:

    python plot.py M E

The command will load and display all the data of completed training runs that are stored in the folder which is specified in `params["output_folder"]` (see `settings.py`).

The evaluation mode `E` are specified in the table below:

| Evaluation mode                | Label |
|--------------------------------|-------|
| VFF operator comparison        | `F`   |
| State-of-the-art comparison    | `S`   |
| Assignment strategy comparison | `A`   |
| Division diversity comparison  | `D`   |

## 7. Rendering

To render episodes of the `Warehouse[N]` or `Battle[N]` domain, set `params["render_pygame"]=True` in `settings.py`.

## 8. References

- [1] T. Phan et al., ["VAST: Value Function Factorization with Variable Agent Sub-Teams"](https://openreview.net/pdf?id=hyJKKIhfxxT), in NeurIPS 2021
