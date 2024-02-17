#!/usr/bin/env bash

for n in 1 2 3 4 
do
    for h in 16 32 64 128
    do
        for enc in 'neuralsde_1_18' 'neuralsde_2_16' 'neuralsde_4_17' 'neuralsde_6_17' 
        do

        python3 sde_interpolation.py --niters 300 --lr 0.001 --batch-size 64 --enc $enc --rec-num-hidden $n --rec-hidden $h  --gen-hidden 64 --latent-dim 32 --dec rnn3 --quantization 0.016  --n 8000  --save 1 --k-iwae 5 --std 0.01 --norm --learn-emb --kl --seed 0 --num-ref-points 64 --dataset physionet --sample-tp 0.9

        python3 sde_interpolation.py --niters 300 --lr 0.001 --batch-size 64 --enc $enc --rec-num-hidden $n --rec-hidden $h  --gen-hidden 64 --latent-dim 32 --dec rnn3 --quantization 0.016  --n 8000  --save 1 --k-iwae 5 --std 0.01 --norm --learn-emb --kl --seed 0 --num-ref-points 64 --dataset physionet --sample-tp 0.8

        python3 sde_interpolation.py --niters 300 --lr 0.001 --batch-size 64 --enc $enc --rec-num-hidden $n --rec-hidden $h  --gen-hidden 64 --latent-dim 32 --dec rnn3 --quantization 0.016  --n 8000  --save 1 --k-iwae 5 --std 0.01 --norm --learn-emb --kl --seed 0 --num-ref-points 64 --dataset physionet --sample-tp 0.7

        python3 sde_interpolation.py --niters 300 --lr 0.001 --batch-size 64 --enc $enc --rec-num-hidden $n --rec-hidden $h  --gen-hidden 64 --latent-dim 32 --dec rnn3 --quantization 0.016  --n 8000  --save 1 --k-iwae 5 --std 0.01 --norm --learn-emb --kl --seed 0 --num-ref-points 64 --dataset physionet --sample-tp 0.6

        python3 sde_interpolation.py --niters 300 --lr 0.001 --batch-size 64 --enc $enc --rec-num-hidden $n --rec-hidden $h  --gen-hidden 64 --latent-dim 32 --dec rnn3 --quantization 0.016  --n 8000  --save 1 --k-iwae 5 --std 0.01 --norm --learn-emb --kl --seed 0 --num-ref-points 64 --dataset physionet --sample-tp 0.5

        done
    done
done
    