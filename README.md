# RL for combustion engine

Environments and scripts for doing reinforcement learning for
injections into different types of engines.

## Running the tests

``` shell
$ conda env create -f environment.yaml
$ nosetests
```

## Getting started

To train an agent, you can run the PPO example:

``` shell
$ cd scripts/ppo-example
$ python ../main.py --fname input.toml
```
