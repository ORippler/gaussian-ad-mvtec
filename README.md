[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/modeling-the-distribution-of-normal-data-in/anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad?p=modeling-the-distribution-of-normal-data-in)

# gaussian-ad-mvtec

This repository provides the code underlying our Publication ["Modeling the Distribution of Normal Data in Pre-Trained Deep Features for Anomaly Detection"](https://arxiv.org/abs/2005.14140) presented at ICPR2020 and its journal extension ["Gaussian Anomaly Detection by Modeling the Distribution of Normal Data in Pretrained Deep Features"](https://ieeexplore.ieee.org/abstract/document/9493210) published at IEEE TIM.

## Updates

* 23.09.2021: Our Journal extension has benen published at IEEE TIM. As the main contribution, we investigate the resistance of our approach to the presence of unlabeled anomalies in the training dataset, and show that our approach can even be applied in the unsupervised setting thanks to the strong prior induced by the Gaussian assumption.

## Introduction & Installation

The code is written in `Python 3.7` using `Pytorch` + `Pytorch-lightning` and was tested under `Ubuntu 18.04`.
All required dependencies can be installed with the `conda` environment as specified in the `environment.yml`.

To setup the environment, execute the following in a terminal:

```
git clone https://github.com/ORippler/gaussian-ad-mvtec.git

cd <path-to-cloned-repo>

conda env create -f environment.yml
```

Apart from the environment, the public [MVTec Anomaly Detection dataset](https://www.mvtec.com/de/unternehmen/forschung/datasets/mvtec-ad/) is also required for running the experiments.
Please download the dataset and export the path to the dataset's base folder as an environemnt variable named `MVTECPATH`.

```
export MVTECPATH=<path-to-mvtec-base-folder>
```

## Running experiments

Individual runs can be launched by calling `python -m src.common.trainer` with a list of corresponding arguments.
Make sure you have created an environment variable `MVTECPATH` and activated the `conda environment` before doing this.

E.g. executing

```
conda activate gaussian-ad-mvtec
export MVTECPATH=<path-to-mvtec-base-folder>
cd <path-to-cloned-repo>

python -m src.common.trainer --model gaussian --category bottle --arch efficientnet-b4 --extract_blocks 0 1 2 3 4 5 6 7 --max_nb_epochs 0
```
would perform a sum-predictor based anomaly detection on the 0-th fold of the bottle category.
Note that `--extract_blocks` are zero indexed in our code, whereas the architectures described in the paper are one-indexed.

Results of a run are stored inside a `lightning_logs` folder created at `--logpath` (default value is `os.cwd()`) with enumerating version count.
Apart from the `metrics.csv` with the results, additional plots/run documentations are available as a tensorboard logs inside each run and can be launched using `tensorboard --logdir <path-to-version>`.

## Recreating paper results

If you wish to recreate paper results, you can run the scripts named `table*.py` located inside `./src/scripts` using e.g.

```
python -m src.scripts.table1 --gpu --logpath <path-to-your-log>
```

`--gpu` enables single-gpu useage, and a `lightning_logs` folder with all individual runs will be created inside `--logpath`.
Should you run out of VRAM (results were generated and verified with 11GB VRAM (GTX 1080 Ti)), you can reduce `--batch-size` inside the scripts.

Recreating a table requires between 20-300 minutes using a GPU.

## Expanding to other datasets

Other datasets can be provided by subclassing the `.src.common.dataset.AnomalyDetectionDataset` inside a python file that is located at `./src/datasets/` and providing the name of the file to the `trainer` via the `--dataset` argument.
Make sure to include a `DATASET = MyDataset` line inside your python file so the trainer knows which class to instantiate.

However, code-base currently only supports 3-D RGB images due to the fact that our method relies on ImageNet pretraining.

## Citation and Contact

If you find our work useful, please consider citing our paper presented at ICPR2020 as well as the journal extension published at IEEE TIM

```
@InProceedings{Rippel2020aModeling,
  author={Rippel, Oliver and Mertens, Patrick and Merhof, Dorit},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)}, 
  title={Modeling the Distribution of Normal Data in Pre-Trained Deep Features for Anomaly Detection}, 
  year={2021},
  volume={},
  number={},
  pages={6726-6733},
  doi={10.1109/ICPR48806.2021.9412109}}
  
 @ARTICLE{9493210,
  author={Rippel, Oliver and Mertens, Patrick and König, Eike and Merhof, Dorit},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Gaussian Anomaly Detection by Modeling the Distribution of Normal Data in Pretrained Deep Features}, 
  year={2021},
  volume={70},
  number={},
  pages={1-13},
  doi={10.1109/TIM.2021.3098381}}
```

If you wish to contact us, you can do so at rippel@lfb.rwth-aachen.de


## License

Copyright (C) 2020 by RWTH Aachen University                      
http://www.rwth-aachen.de                                             
                                                                         
License:                                                                                                                                       
This software is dual-licensed under:                                 
• Commercial license (please contact: lfb@lfb.rwth-aachen.de)         
• AGPL (GNU Affero General Public License) open source license 
