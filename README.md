# Alpha Zero for Gobang

An experimental implementation of Alpha Zero Algorithm for game [Gomoku](https://en.wikipedia.org/wiki/Gomoku) (connect5, 5-in-a-row, Gobang).

Originally from suragnair's [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) project, with more efficient asynchronized and parallelized sampling & training implementation for easy scaling up.

The code is based on multiprocessing package, with 5 different type processes:
- Sampler, that host and run a batch of MCTS, send state query batch to Evaluator and recieve feedbacks, send collected training samples to Trainer.
- Evaluator, that host a copy of neural network and process queries from sampler.
- Trainer, that host a copy of neural network and do (data-parallel) parameter update once collect enough samples from Sampler.
- Arena, that run games between candidates and push latest models to other component.
- Controller, that moniter and control the overall status.  

The number of these components can be customized and allocted according to computational resources, allowing scaling to larger model size and training procedure. 

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

### Result
The code is tested on a 32 core CPU, 256 GB RAM, and 8 GPU cards machine, with gcc 12, python 3.6.8, and pytorch 1.7.1. We provide trained models for boardsize 9 and 15 at [this link](https://pan.baidu.com/s/1-XLuWbVDDEFtIRwK8-dEvA?pwd=1s2g). Checkpoint file name are formatted by boardsize(n), res-block num(b), channel num(c), trained iterations(i), and save date(d).

On boardsize 9, the game is very likely to end in a draw. On boardsize 15, the agent plays strong black (first to play), but very weak white. Perhaps this can be attributed to the strong advantage for the first player when game is unrestricted, and agent tends to play non-sense move when the game is hopeless. Below is a match between the agent (black) and the author (white).

<a href="https://ibb.co/P9JjWGM"><img src="https://i.ibb.co/SdHymwc/wechat-20230802102237.png" alt="wechat-20230802102237" border="0" width=60%></a>

Zhiyuan