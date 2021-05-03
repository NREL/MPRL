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

## Citing
```
@article{Henrydefrahan2021,
  title={Deep reinforcement learning for dynamic control of fuel injection timing in multi-pulse compression ignition engines},
  author={Henry de Frahan, Marc T and Wimer, Nicholas T and Yellapantula, Shashank and Grout, Ray W},
  journal={International Journal of Engine Research},
  volume={-},
  pages={-},
  year={2021},
  publisher={Sage}
}
```
