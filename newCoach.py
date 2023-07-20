import logging
from random import shuffle

import numpy as np
from tqdm import tqdm
from multiprocessing import Lock, RLock, Queue, Process, Value
import time
from circular_dict import CircularDict
import pickle, os

from Arena import Arena
from MCTS import batch_MCTS

log = logging.getLogger(__name__)

def Sampler(identifier, q_job, q_ans, qdata, v_model, game, args, lock):
    num_workers = args.mcts_per_sampler
    mem_limit_bytes = int(args.available_mem_gb * 0.95 / args.sampler_num / 3 * 1024 * 1024 * 1024)
    shared_Ps = CircularDict(maxsize_bytes=mem_limit_bytes)
    shared_Es = CircularDict(maxsize_bytes=mem_limit_bytes)
    shared_Vs = CircularDict(maxsize_bytes=mem_limit_bytes)
    query_buffer = []
    worker_pool = {i: batch_MCTS(game,args,shared_Ps,shared_Es,shared_Vs,query_buffer,i) for i in range(num_workers)}
    trainExamples = []
    model_version = 0
    game_counter = 0
    game_counter_old = 0
    start_time = time.time()
    gpu_time = 0
    t_bar = tqdm(total=100, desc=f'Sampler {identifier:{2}}', position=identifier+3, lock_args=(True, args.tqdm_wait_time))
    t_bar.set_lock(lock)

    while 1:
        # sample
        for _ in range(args.numMCTSSims):
            # extend
            for i in range(num_workers):
                worker_pool[i].extend()
            
            # query network
            _start_time = time.time()
            query_index, query_content, query_state_string = zip(*query_buffer)
            q_job.put(np.array(query_content))
            query_buffer.clear()
            pi, v = q_ans.get()
            _end_time = time.time()
            gpu_time += _end_time - _start_time
            
            # set result
            for j,s in enumerate(query_state_string):
                # Set Ps and Vs
                shared_Ps[s] = pi[j]
                if s not in shared_Vs:
                    valids = game.getValidMoves(query_content[j], 1)
                    shared_Vs[s] = valids
                else:
                    valids = shared_Vs[s]
                shared_Ps[s] = shared_Ps[s] * valids
                sum_Ps_s = np.sum(shared_Ps[s])
                if sum_Ps_s > 0:
                    shared_Ps[s] /= sum_Ps_s  # renormalize
                else:  
                    log.error("All valid moves were masked, doing a workaround.")
                    shared_Ps[s] = shared_Ps[s] + valids
                    shared_Ps[s] /= np.sum(shared_Ps[s])
                
                # Set values
                worker_pool[query_index[j]].current_value = -v[j]

            # backprop
            for i in range(num_workers):
                worker_pool[i].backprop()

        # take action
        for i in range(num_workers):
            mcts = worker_pool[i]
            canonicalBoard = game.getCanonicalForm(mcts.board, mcts.player)
            temp = int(mcts.episodeStep < args.tempThreshold)
            s = game.stringRepresentation(canonicalBoard)
            counts = [mcts.Nsa[s][a] if s in mcts.Nsa else 0 for a in range(game.getActionSize())]

            if temp == 0:
                bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
                bestA = np.random.choice(bestAs)
                probs = [0] * len(counts)
                probs[bestA] = 1
                pi = probs
            else:
                counts = [x ** (1. / temp) for x in counts]
                counts_sum = float(sum(counts))
                pi = [x / counts_sum for x in counts]
            sym = game.getSymmetries(canonicalBoard, pi)
            mcts.game_record += [(b, p, mcts.player) for b, p in sym]

            action = np.random.choice(len(pi), p=pi)
            mcts.board, mcts.player = game.getNextState(mcts.board, mcts.player, action)
            mcts.episodeStep += 1

            r = game.getGameEnded(mcts.board, mcts.player)
            if r != 0:
                trainExamples += [(x[0], x[1], r * ((-1) ** (x[1] != mcts.player))) for x in mcts.game_record]
                worker_pool[i].reset()
                game_counter += 1

        # Send Data
        if len(trainExamples)>256:
            # log.debug(f"Sampler: send to trainer. current result_pool size {len(trainExamples)}")
            qdata.put(trainExamples)
            trainExamples = []

        # reset buffer upon new model
        if v_model.value > model_version:
            model_version = v_model.value
            log.debug(f"Sampler: reset mcts buffer for model {model_version}")
            # Reset all mcts? Todo: update Ps with new model?
            shared_Ps.clear() 
            # for i in range(num_workers):
            #     worker_pool[i].reset()

        # Update state
        gpu_ratio = gpu_time/(time.time()-start_time)
        random_search_depth = np.mean([worker_pool[i].total_search_depth/worker_pool[i].search_count for i in range(num_workers)])
        t_bar.set_postfix(game=game_counter,gpu=gpu_ratio,sd=random_search_depth)
        t_bar.update(game_counter-game_counter_old)
        if t_bar.n >= 100:
            t_bar.reset()
        game_counter_old = game_counter

def Evaluator(sampler_pool, v_model, gpu_id, game, nn, args, lock):
    ''' This function is the Evaluator process, that:
            - evluate queries from sampler
            - check new models from trainer
    '''
    nnet = nn(game, gpu_id)
    model_version = 0

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    while 1:
        # load new model once avaliable
        if v_model.value > model_version:
            model_version = v_model.value
            nnet.load_checkpoint(folder=args.checkpoint, filename=f"checkpoint_{model_version}.pth.tar")
            log.debug(f"Evaluator: pull new model {model_version}")

        # evaluate queries
        for i in sampler_pool:
            q_job = sampler_pool[i][0]
            q_ans = sampler_pool[i][1]
            if q_job.qsize()>0:
                job = q_job.get()
                ans = nnet.predict(np.array(job))
                q_ans.put(ans)

def Trainer(q_data, v_model, gpu_id, game, nn, args, lock):
    ''' This function is the Trainer process, that:
            - collect data from sampler
            - train new models and push latest one to evaluator
    '''
    episode_size = 100000

    t_status = tqdm(total=args.numIters, desc='Status', position=0, lock_args=(True, args.tqdm_wait_time))
    t_status.set_lock(lock)
    t_train = tqdm(total=1, desc='Training', position=1, lock_args=(True, args.tqdm_wait_time))
    t_train.set_lock(lock)
    t_sample = tqdm(total=episode_size, desc="Sampling", position=2, lock_args=(True, args.tqdm_wait_time))
    t_sample.set_lock(lock)

    nnet = nn(game, gpu_id, t_train)
    episodeHistory = []
    sample_buffer = []
    data_update_time = time.time()
    last_train_time = data_update_time + 1
    train_iter_counter = 1
    start_time = time.time()

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        if len(args.load_folder_file) > 2:
            with open(os.path.join(args.load_folder_file[0], args.load_folder_file[2]), 'rb') as f:
                episodeHistory = pickle.load(f)
    
    while 1:
        if q_data.qsize()>0:
            for _ in range(q_data.qsize()):
                data = q_data.get()
                sample_buffer += data
                log.debug(f"Trainer: recieved from sampler. data size {len(data)}, buffer size{len(sample_buffer)}")
                t_sample.update(len(data))
                if len(sample_buffer) > episode_size:
                    episodeHistory.append(sample_buffer)
                    data_update_time = time.time()
                    sample_buffer = []
                    t_sample.reset()
                if len(episodeHistory) > args.numItersForTrainExamplesHistory:
                    log.debug(f"Trainer: Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(episodeHistory)}")
                    episodeHistory.pop(0)

        if last_train_time < data_update_time:
            trainExamples = []
            for e in episodeHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)
            nnet.train(trainExamples)
            last_train_time = time.time()
            _model_name = 'checkpoint_' + str(int(last_train_time)) + '.pth.tar'
            _data_name = 'checkpoint_' + str(int(last_train_time)) + '.data.pth.tar'
            nnet.save_checkpoint(folder=args.checkpoint, filename=_model_name)
            with open(os.path.join(args.checkpoint, _data_name), 'wb') as f:
                pickle.dump(episodeHistory, f)
            v_model.value = int(last_train_time)
            log.debug(f"Trainer: push model {v_model.value}")
            train_iter_counter += 1
            t_status.update(1)

        t_status.set_postfix(iter=train_iter_counter, model=str(int(last_train_time)), datasize=sum([len(x) for x in episodeHistory]))
        time.sleep(0.1)

class mpCoach():
    def __init__(self, game, nnet, args):
        self.game = game
        self.nn = nnet # class of NNwraper, NOT a instance of it
        self.args = args
        self.global_lock = RLock()

    def learn(self):
        q_data = Queue()
        v_model = Value('i', -1)
        sampler_num = self.args.sampler_num
        evaluator_num = self.args.gpu_num - 1
        sampler_pool = {}

        for i in range(sampler_num):
            q_job = Queue()
            q_ans = Queue()
            sampler = Process(target=Sampler, args=(i, q_job, q_ans, q_data, v_model, self.game, self.args, self.global_lock))
            sampler.start()
            sampler_pool[i] = [q_job, q_ans]
        
        for i in range(evaluator_num):
            sampler_pool_subset = {k: sampler_pool[k] for j,k in enumerate(sampler_pool) if j%evaluator_num==i}
            evaluator = Process(target=Evaluator, args=(sampler_pool_subset, v_model, 1+i, self.game, self.nn, self.args, self.global_lock))
            evaluator.start()
        
        trainer = Process(target=Trainer, args=(q_data, v_model, 0, self.game, self.nn, self.args, self.global_lock))
        trainer.start()
        trainer.join()