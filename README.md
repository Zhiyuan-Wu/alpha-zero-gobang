# Alpha Zero for Gobang

An implementation of Alpha Zero Algorithm for game Gobang (connect5, 5-in-a-row, Gomoku).

Originally from suragnair's [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) project, with more efficient asynchronized and parallelized sampling & training implementation for easy scaling up.

The code is based on multiprocessing package, with 5 different type processes:
- Sampler, that host and run a batch of MCTS, send state query batch to Evaluator and recieve feedbacks, send collected training samples to Trainer.
- Evaluator, that host a copy of neural network and process queries from sampler.
- Trainer, that host a copy of neural network and do (data-parallel) parameter update once collect enough samples from Sampler.
- (Todo) Arena, that run games between candidates to and push latest models to other component.
- Controller, that moniter and control the overall status.  

The number of these components can be customized and allocted according to computational resources, allowing scaling to larger model size and training procedure. The code is tested on a 32 core CPU, 256 GB RAM, and 8 GPU cards machine. 

Zhiyuan