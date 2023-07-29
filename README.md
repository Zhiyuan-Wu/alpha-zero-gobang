# Alpha Zero for Gobang

An implementation of Alpha Zero Algorithm for game Gobang (connect5, 5-in-a-row, Gomoku).

Originally from suragnair's [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) project, with more efficient asynchronized and parallelized sampling & training implementation for easy scaling up.

The code is based on multiprocessing package, with 5 different type processes:
- Sampler, that host and run a batch of MCTS, send state query batch to Evaluator and recieve feedbacks, send collected training samples to Trainer.
- Evaluator, that host a copy of neural network and process queries from sampler.
- Trainer, that host a copy of neural network and do (data-parallel) parameter update once collect enough samples from Sampler.
- Arena, that run games between candidates and push latest models to other component.
- Controller, that moniter and control the overall status.  

The number of these components can be customized and allocted according to computational resources, allowing scaling to larger model size and training procedure. The code is tested on a 32 core CPU, 256 GB RAM, and 8 GPU cards machine, with gcc 12, python 3.6.8, and pytorch 1.7.1.

### Usage
To train the model, first config parameters in `main.py`, then:
```bash
cd cfun
python setup.py build_ext --inplace
cd ..
python main.py
```

To play with the trained model, first config model in `gtp.py`, then use any GUI that support GTP (e.g., [Sabaki](https://github.com/SabakiHQ/Sabaki)) with following setting:
```
command: python
arguments: path/to/gtp.py --game_size 15 --model_path path/to/checkpoint.pth --numMCTSSims 1000
```

Zhiyuan