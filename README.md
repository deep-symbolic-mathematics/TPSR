# TPSR

Official Implementation of **[Transformer-based Planning for Symbolic Regression](https://arxiv.org/abs/2303.06833)** (accepted at NeurIPS 2023). 


## Overview
In this paper, we introduce **TPSR**, a novel transformer-based planning framework for symbolic regression by leveraging priors of large-scale pretrained models and incorporating lookahead planning. TPSR incorporates Monte Carlo Tree Search (MCTS) into the transformer decoding process of symbolic regression models. Unlike conventional decoding strategies, TPSR enables the integration of non-differentiable feedback, such as fitting accuracy and complexity, as external sources of knowledge into the transformer-based equation generation process.

<p align="center">
<img src="./images/Media13_Final.gif" width="100%" /> 
 <br>
<b>TPSR uncovering the governing symbolic mathematics of data, providing enhanced extrapolation capabilities.</b>
</p>


## Installation

To run the code, create a conda ``Python 3.7`` environment and install the dependencies by running the following command.

```
conda create --name tpsr python=3.7
pip install -r requirements.txt
```


## Preperation: Data and Pre-trained Backbone Models 
<!-- ### Download Manually -->
1. Download Pre-trained Models:
    * **End-to-End (E2E) SR Transformer model** is available [here](https://dl.fbaipublicfiles.com/symbolicregression/model1.pt).
    * **NeSymReS model** is available [here](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales).
      
After downloading, extract both models to this directory. They should be located under the `pretrained_models/` folder.

2. Download Benchmark Datasets:
    * **Feynman** equations are [here](https://space.mit.edu/home/tegmark/aifeynman.html)
    * **PMLB** regression datasets are also [here](https://github.com/EpistasisLab/pmlb/tree/master/datasets). Data points of PMLB datasets are used in the [SRBench: A Living Benchmark for Symbolic Regression)](https://github.com/cavalab/srbench), containing three data groups: **Feynman**, **Strogatz**, and **Black-box**.
      
Extract the datasets to this directory, Feynman datasets should be in `datasets/feynman/`, and PMLB datasets should be in `datasets/pmlb/`. 


## Run
We have created `run.sh` script to execute Transformer-based Planning for Equation Generation based on the a reward defined in `reward.py` with the combination of equation's fitting accuracy and complexity. To run the script for different datasets, configure the following parameters:

|   **Parameters**  |                                              **Description**                                             |       **Example Values**       |
|:-----------------:|:--------------------------------------------------------------------------------------------------------:|:------------------------------:|
| `eval_in_domain`    | Evaluate Backbone Pre-trained Model on In-Domain Dataset (Yes/No)                                | True/False      |
| `eval_mcts_in_domain`    | Evaluae TPSR on In-Domain Dataset (Yes/No)                                | True/False      |
| `eval_on_pmlb`        | Evaluae Backbone Pre-trained Model on PMLB (Yes/No)                                                                     | True/False |
| `eval_mcts_on_pmlb`    | Evaluae TPSR on PMLB (Yes/No)                                  | True/False       |
| `horizon`    | Horizon of Lookahead Planning (MaxLen of Equations)                                 | 200      |
| `rollout`    | Number of Rollouts ($r$) in TPSR                                 | 3      |
| `num_beams`    | Beam Size ($b$) in TPSR's Evaluation Step to Simulate Completed Equations                                | 1      |
| `width`    |  Top-k ($k$) in TPSR's Expansion Step to Expand Tree Width                                | 3     |
| `no_seq_cache`    | Use Sequence Caching (Yes/No)                                 | False     |
| `no_prefix_cache`    | Use Top-k Caching (Yes/No)                                   | False     |
| `ucb_constant`    | Exploration Weight in UCB                                 | 1.0     |
| `uct_alg`    | UCT Algorithm $\in$ {uct, p_uct, var_p_uct}                                 | uct     |
| `max_input_points`    | Maximum Input Points Observed by Pre-trained Model ($N$)                                 | 200     |
| `max_number_bags`    | Maximum Number of Bags for Input Points ($B$)                                | 10     |
| `pmlb_data_type`    | PMLB Data Group $\in$ {feynman, strogatz, black-box}                                 | feynman     |
| `target_noise`    | Target Noise added to y_to_fit in PMLB                                | 0.0     |
| `beam_type`    | Decoding Type for Pre-trained Models $\in$ {search, sampling}                                | sampling     |
| `beam_size`    | Decoding Size ($s$) for Pre-trained Models (Beam Size, or Sampling Size)                                | 10     |
| `n_trees_to_refine`    | Number of Refinements in Decodings $\in$ {1,...,$s$}                                | 10     |
| `prediction_sigmas`    | Sigmas of Extrapolation Eval Data Sampling (In-domain)                                | 1,2,4,8,16     |
| `eval_input_length_modulo`    | Number of Eval Points (In-domain). Set to 50 Yields $N_{test}=[50,100,150,200]$ per Extrapolation Range.                               | 50     |


## Run - PMLB Datasets (Feynman/ Strogatz/ Blackbox)
Pre-trained E2E Model (Sampling):
```
python run.py --eval_on_pmlb True \
                   --pmlb_data_type feynman \
                   --target_noise 0.0 \
                   --beam_type sampling \
                   --beam_size 10 \
                   --n_trees_to_refine 10 \
                   --max_input_points 200 \
                   --max_number_bags 10 \
                   --save_results True
```

Pre-trained E2E Model (Beam Search):
```
python run.py --eval_on_pmlb True \
                   --pmlb_data_type feynman \
                   --target_noise 0.0 \
                   --beam_type search \
                   --beam_size 10 \
                   --n_trees_to_refine 10 \
                   --max_input_points 200 \
                   --max_number_bags 10 \
                   --save_results True
```

Transformer-based Planning with E2E Backbone:
```
python run.py --eval_mcts_on_pmlb True \
                   --pmlb_data_type feynman \
                   --target_noise 0.0 \
                   --lam 0.1 \
                   --horizon 200 \
                   --width 3 \
                   --num_beams 1 \
                   --rollout 3 \
                   --no_seq_cache False \
                   --no_prefix_cache True \
                   --max_input_points 200 \
                   --max_number_bags 10 \
                   --save_results True
```
For running the code on Strogatz or Black-box datasets, simply adjust the `pmlb_data_type` parameter to either `strogatz` or `blackbox`. The commands provided above are set for the Feynman datasets. You can also modify the `target_noise` and other parameters to suit your experiments. Running each command saves the results for all datasets and metrics in a `.csv` file. 




## Run - In-Domain Datasets
**In-Domain** datasets are generated, following the validation data gneration protocol suggested in [E2E](https://arxiv.org/pdf/2204.10532.pdf). For details, refer to the `generate_datapoints` function [here](./symbolicregression/envs/generators.py). You can also modify data generation parameters [here](./symbolicregression/envs/environment.py). For example, you can adjust parameters like `prediction_sigmas` to control extrapolation. A sigma `1` aligns with the training data range, while `>1` is for extrapolation ranges.  The In-domain validation datasets are generated on-the-fly. For consistent evaluations across models, consider setting a fixed seed.


Pre-trained E2E Model (Sampling):
```
python run.py --eval_in_domain True \
                   --beam_type sampling \
                   --beam_size 10 \
                   --n_trees_to_refine 10 \
                   --max_input_points 200 \
                   --eval_input_length_modulo 50 \
                   --prediction_sigmas 1,2,4,8,16 \
                   --save_results True
```

Pre-trained E2E Model (Beam Search):
```
python run.py --eval_in_domain True \
                   --beam_type search \
                   --beam_size 10 \
                   --n_trees_to_refine 10 \
                   --max_input_points 200 \
                   --eval_input_length_modulo 50 \
                   --prediction_sigmas 1,2,4,8,16 \
                   --save_results True
```

Transformer-based Planning with E2E Backbone:
```
python run.py --eval_mcts_in_domain True \
                   --lam 0.1 \
                   --horizon 200 \
                   --width 3 \
                   --num_beams 1 \
                   --rollout 3 \
                   --no_seq_cache False \
                   --no_prefix_cache True \
                   --max_input_points 200 \
                   --eval_input_length_modulo 50 \
                   --prediction_sigmas 1,2,4,8,16 \
                   --save_results True \
                   --debug
```

## Runs With NeSymReS Backbone
To be released soon.


## Demo
We have also included a small demo that runs TPSR with E2E backbone on your dataset. You can play with it [here](./tpsr_demo.py) 


## Citation
If you find the paper or the source code useful to your projects, please cite the following:
<pre>
@article{tpsr2023,
  title={Transformer-based Planning for Symbolic Regression},
  author={Shojaee, Parshin and Meidani, Kazem and Farimani, Amir Barati and Reddy, Chandan K},
  journal={arXiv preprint arXiv:2303.06833},
  year={2023}
}
</pre>



## License 
This repository is licensed under MIT licence.



This work is built on top of other open source projects: including: [End-to-End Symbolic Regression with Transformers (E2E)](https://github.com/facebookresearch/symbolicregression), [Neural Symbolic Regression that scales (NeSymReS)](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales), [Dyna Gym](https://github.com/SuReLI/dyna-gym), and [transformers](https://github.com/huggingface/transformers). We thank the original contributors of these works for open-sourcing their valuable source codes. 

## Contact Us
For any questions or issues, you are welcome to open an issue in this repo, or contact us at parshinshojaee@vt.edu, and mmeidani@andrew.cmu.edu .
