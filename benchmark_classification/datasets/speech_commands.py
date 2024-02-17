import os
import pathlib
import urllib.request
import tarfile
import torch
import torchaudio

from . import common

here = pathlib.Path(__file__).resolve().parent


def download():
    base_base_loc = here / 'data'
    base_loc = base_base_loc / 'SpeechCommands'
    loc = base_loc / 'speech_commands.tar.gz'
    if os.path.exists(loc):
        return
    if not os.path.exists(base_base_loc):
        os.mkdir(base_base_loc)
    if not os.path.exists(base_loc):
        os.mkdir(base_loc)
    urllib.request.urlretrieve('http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz', loc)
    with tarfile.open(loc, 'r') as f:
        f.extractall(base_loc)


def _process_data(intensity_data):
    base_loc = here / 'data' / 'SpeechCommands'
    X = torch.empty(34975, 16000, 1)
    y = torch.empty(34975, dtype=torch.long)

    batch_index = 0
    y_index = 0
    for foldername in ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'):
        loc = base_loc / foldername
        for filename in os.listdir(loc):
            # audio, _ = torchaudio.load_wav(loc / filename, channels_first=False,
            #                                normalization=False)  # for forward compatbility if they fix it
            audio, _ = torchaudio.load(loc / filename, channels_first=False,
                                       normalize=True)  # for forward compatbility if they fix it
            audio = audio / 2 ** 10  # Normalization argument doesn't seem to work so we do it manually.

            # A few samples are shorter than the full length; for simplicity we discard them.
            if len(audio) != 16000:
                continue

            X[batch_index] = audio
            y[batch_index] = y_index
            batch_index += 1
        y_index += 1
    assert batch_index == 34975, "batch_index is {}".format(batch_index)

    X = torchaudio.transforms.MFCC(log_mels=True, n_mfcc=20,
                                   melkwargs=dict(n_fft=200, hop_length=100, n_mels=128))(X.squeeze(-1)).transpose(1, 2).detach()
    print(X.shape)
    # X is of shape (batch=34975, length=161, channels=20)

    times = torch.linspace(0, X.size(1) - 1, X.size(1))
    final_index = torch.tensor(X.size(1) - 1).repeat(X.size(0))

    (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, _) = common.preprocess_data(times, X, y, final_index, append_times=True,
                                                   append_intensity=intensity_data)

    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index)


def get_data(intensity_data, batch_size):
    base_base_loc = here / 'processed_data'
    loc = base_base_loc / ('speech_commands_with_mels' + ('_intensity' if intensity_data else ''))
    if os.path.exists(loc):
        tensors = common.load_data(loc)
        times = tensors['times']
        train_coeffs = tensors['train_coeffs']
        val_coeffs = tensors['val_coeffs']
        test_coeffs = tensors['test_coeffs']
        train_y = tensors['train_y']
        val_y = tensors['val_y']
        test_y = tensors['test_y']
        train_final_index = tensors['train_final_index']
        val_final_index = tensors['val_final_index']
        test_final_index = tensors['test_final_index']
    else:
        download()
        (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
         test_final_index) = _process_data(intensity_data)
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        common.save_data(loc, times=times,
                         train_coeffs=train_coeffs, val_coeffs=val_coeffs, test_coeffs=test_coeffs,
                         train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                         val_final_index=val_final_index, test_final_index=test_final_index)

    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(times, train_coeffs, val_coeffs,
                                                                                test_coeffs, train_y, val_y, test_y,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, 'cpu',
                                                                                batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader
