# pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

from random import SystemRandom
import models
import utils

import torchcde
from torch_ists import ists_layer

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--rec-num-hidden', type=int, default=1)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--gen-hidden', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--k-iwae', type=int, default=10)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='neuralsde_0_18')
parser.add_argument('--dec', type=str, default='rnn3')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--quantization', type=float, default=0.016,
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true',
                    help="Include binary classification loss")
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dec-num-heads', type=int, default=1)
parser.add_argument('--length', type=int, default=20)
parser.add_argument('--num-ref-points', type=int, default=64)
parser.add_argument('--dataset', type=str, default='toy')
parser.add_argument('--enc-rnn', action='store_false')
parser.add_argument('--dec-rnn', action='store_false')
parser.add_argument('--sample-tp', type=float, default=1.0)
parser.add_argument('--only-periodic', type=str, default=None)
parser.add_argument('--dropout', type=float, default=0.0)
args = parser.parse_args()


if __name__ == '__main__':
    experiment_id = int(SystemRandom().random() * 100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'toy':
        data_obj = utils.kernel_smoother_data_gen(args, alpha=100., seed=0)
    elif args.dataset == 'physionet':
        data_obj = utils.get_physionet_data(args, 'cpu', args.quantization)

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    dim = data_obj["input_dim"]

    # model
    # if args.enc == 'enc_rnn3':
    #     rec = models.enc_rnn3(
    #         dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, 
    #         args.rec_hidden, 128, learn_emb=args.learn_emb).to(device)
    # elif args.enc == 'mtan_rnn':
    #     rec = models.enc_mtan_rnn(
    #         dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden, 
    #         embed_time=128, learn_emb=args.learn_emb, num_heads=args.enc_num_heads).to(device)
   

    input_dim = 41
    seq_len = args.embed_time

    num_layers = 1
    num_hidden_layers = args.rec_num_hidden

    hidden_dim = args.rec_hidden
    hidden_hidden_dim = args.rec_hidden
    out_dim = args.latent_dim*2

    model_name = args.enc
    print(model_name)
    
    rec_sde = ists_layer(model_name=model_name, input_dim=input_dim, seq_len=seq_len,
                         hidden_dim=hidden_dim, hidden_hidden_dim=hidden_hidden_dim, 
                         num_layers=num_layers, num_hidden_layers=num_hidden_layers, 
                         bidirectional=False, dropout=0.1, use_intensity=False, 
                         method='euler', file=None, device=device).to(device) 
    rec_out = nn.Linear(hidden_dim, out_dim).to(device)

    class rec_combined(nn.Module):
        def __init__(self, rec_sde, rec_out):
            super().__init__()
            self.rec_sde = rec_sde
            self.rec_out = rec_out

        def forward(self, seq, coeffs):
            out = rec_sde(seq, coeffs)
            return rec_out(out[0])

    rec = rec_combined(rec_sde, rec_out)

    if args.dec == 'rnn3':
        dec = models.dec_rnn3(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, 
            args.gen_hidden, 128, learn_emb=args.learn_emb).to(device)
    elif args.dec == 'mtan_rnn':
        dec = models.dec_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.gen_hidden, 
            embed_time=128, learn_emb=args.learn_emb, num_heads=args.dec_num_heads).to(device)


    params = (list(dec.parameters()) + list(rec.parameters()))
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec))
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 1))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 3))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 10))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 20))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 30))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 50))

    for itr in range(1, args.niters + 1):
        train_loss = 0
        train_n = 0
        avg_reconst, avg_kl, mse = 0, 0, 0
        if args.kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))
        else:
            kl_coef = 1

        for train_batch in train_loader:
            train_batch = train_batch.to(device)
            batch_len = train_batch.shape[0]
            observed_data = train_batch[:, :, :dim]
            observed_mask = train_batch[:, :, dim:2 * dim]
            observed_tp = train_batch[:, :, -1]
            if args.sample_tp and args.sample_tp < 1:
                subsampled_data, subsampled_tp, subsampled_mask = utils.subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            else:
                subsampled_data, subsampled_tp, subsampled_mask = \
                    observed_data, observed_tp, observed_mask
            # out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)

            subsampled_data, subsampled_mask = subsampled_data.to(device), subsampled_mask.to(device)
            idx_list = (subsampled_tp * args.num_ref_points - 1).long().to(device)

            formatted_data = torch.zeros(subsampled_data.shape[0], args.num_ref_points, subsampled_data.shape[2], device=device)
            formatted_mask = torch.zeros(subsampled_data.shape[0], args.num_ref_points, subsampled_data.shape[2], device=device)

            formatted_data[torch.arange(formatted_data.shape[0])[:,None], idx_list] = subsampled_data
            formatted_mask[torch.arange(formatted_mask.shape[0])[:,None], idx_list] = subsampled_mask            
            
            X_missing, X_mask = formatted_data, formatted_mask
            times = torch.linspace(0, 1, X_missing.shape[1]).to(X_missing.device)
            values_T = torch.cat([times.repeat((X_missing.shape[0],1)).unsqueeze(-1), X_missing], dim=-1)
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(values_T, times)

            seq = torch.stack([
                torch.nan_to_num(X_missing, 0),
                X_mask, X_mask
            ], dim=1).to(device)
            out = rec(seq, coeffs)                

            qz0_mean = out[:, :, :args.latent_dim]
            qz0_logvar = out[:, :, args.latent_dim:]
            # epsilon = torch.randn(qz0_mean.size()).to(device)
            epsilon = torch.randn(
                args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            ).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            pred_x = dec(
                z0,
                observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1])
            )
            # nsample, batch, seqlen, dim
            pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2])
            # compute loss
            logpx, analytic_kl = utils.compute_losses(
                dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
            loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(args.k_iwae))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_len
            train_n += batch_len
            avg_reconst += torch.mean(logpx) * batch_len
            avg_kl += torch.mean(analytic_kl) * batch_len
            mse += utils.mean_squared_error(
                observed_data, pred_x.mean(0), observed_mask) * batch_len
        print('Iter: {}, avg elbo: {:.4f}, avg reconst: {:.4f}, avg kl: {:.4f}, mse: {:.6f}'
            .format(itr, train_loss / train_n, -avg_reconst / train_n, avg_kl / train_n, mse / train_n))

        scheduler.step() ## step scheduler

        if itr % 20 == 0:
            test_mse = utils.evaluate(dim, rec, dec, test_loader, args, 1)
            print('Test Mean Squared Error', test_mse)
        if itr % 20 == 0 and args.save:
            torch.save({
                'args': args,
                'epoch': itr,
                # 'rec_state_dict': rec.state_dict(),
                # 'dec_state_dict': dec.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                'loss': -loss,
                'out': [itr, train_loss / train_n, -avg_reconst / train_n, avg_kl / train_n, mse / train_n, test_mse]
            }, 'interpolation/' + 
                args.dataset + '_' + args.enc + '_' + args.dec + '_' + str(itr) + '_' + 
                str(experiment_id) + '.h5')
