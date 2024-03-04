# bo-explorations

A repository for explorations on different Bayesian optimization setups using the [BoTorch](https://github.com/pytorch/botorch) library

To run the code, I'm typically updating a conda environment that, on thhe first time, can be installed using the following commands:

`mamba create -n botorch_mar2024 pytorch torchvision torchaudio pytorch-cuda=11.8 python==3.11 -c pytorch -c nvidia`

`mamba install botorch matplotlib seaborn -c pytorch -c gpytorch -c conda-forge`

`mamba update -c conda-forge ffmpeg`
