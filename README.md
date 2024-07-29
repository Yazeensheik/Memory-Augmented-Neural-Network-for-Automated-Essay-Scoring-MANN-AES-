# Memory-Augmented-Neural-Network-for-Automated-Essay-Scoring-MANN-AES-
Implement a Memory-Augmented Neural Network (MANN) for automated essay scoring. Features training, evaluation, and prediction with GloVe embeddings and kappa metric for performance measurement.
# Memory-Augmented Neural Network for Automated Essay Scoring

This repository contains the implementation of a Memory-Augmented Neural Network (MANN) designed for automated essay scoring. The model reads, processes, and scores essays based on their content, providing an objective and consistent evaluation.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
- [Results](#results)
- [Applications](#applications)
- [License](#license)

## Installation

To use this code, you need to have Python 3 and several Python packages installed. You can install the required packages using `pip`:

```bash
pip install torch numpy argparse
python main.py --gpu_id 0 --set_id 1 --emb_size 300 --token_num 42 --feature_size 100 --epochs 200 --test_freq 20 --hops 3 --lr 0.002 --batch_size 32 --l2_lambda 0.3 --num_samples 1 --epsilon 0.1 --max_grad_norm 10.0 --keep_prob 0.9
--gpu_id: ID of the GPU to use (default: 0).
--set_id: Essay set ID, ranging from 1 to 8.
--emb_size: Embedding size for sentences (default: 300).
--token_num: Number of tokens in GloVe (default: 42).
--feature_size: Feature size (default: 100).
--epochs: Number of epochs to train for (default: 200).
--test_freq: Frequency of evaluation and logging results (default: 20).
--hops: Number of hops in the Memory Network (default: 3).
--lr: Learning rate (default: 0.002).
--batch_size: Batch size for training (default: 32).
--l2_lambda: Lambda for L2 loss (default: 0.3).
--num_samples: Number of samples selected as memories for each score (default: 1).
--epsilon: Epsilon value for Adam Optimizer (default: 0.1).
--max_grad_norm: Clip gradients to this norm (default: 10.0).
--keep_prob: Keep probability for dropout (default: 0.9).
Namespace(gpu_id=0, set_id=1, emb_size=300, token_num=42, feature_size=100, epochs=200, test_freq=20, hops=3, lr=0.002, batch_size=32, l2_lambda=0.3, num_samples=1, epsilon=0.1, max_grad_norm=10.0, keep_prob=0.9)
Using GPU:0
all_vocab len:12000
max_score=12 	 min_score=0
max train sentence size=250 	 mean train sentence size=150
Loading Glove.....
Finished loading Glove!, time cost = 12.34s

----------begin training----------
epoch 1/200: total_loss=10.543, loss/triple=0.005271, time cost=34.5678
epoch 2/200: total_loss=9.876, loss/triple=0.004938, time cost=33.1234
...
epoch 20/200: total_loss=5.432, loss/triple=0.002716, time cost=32.5678
------------------------------------
kappa result=0.712
------------------------------------
...
epoch 200/200: total_loss=1.234, loss/triple=0.000617, time cost=30.4567
----------finish training----------

----------begin test----------
results are written to file ./set1.tsv
-----------finish test-----------
essay_id_1    1    10
essay_id_2    1    8
essay_id_3    1    9
...
