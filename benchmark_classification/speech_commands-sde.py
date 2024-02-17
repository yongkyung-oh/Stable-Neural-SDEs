import os
import torch

# CUDA for PyTorch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

import common_sde as common
import datasets


def main(device='cuda', max_epochs=200, *,                                        # training parameters
         model_name, hidden_channels, hidden_hidden_channels, num_hidden_layers,  # model parameters
         dry_run=False,
         **kwargs):                                                               # kwargs passed on to cdeint

    batch_size = 1024
    lr = 1e-3

    intensity_data = True if model_name in ('odernn', 'dt', 'decay') else False
    times, train_dataloader, val_dataloader, test_dataloader = datasets.speech_commands.get_data(intensity_data,
                                                                                                 batch_size)
    input_channels = 1 + (1 + intensity_data) * 20

    make_model = common.make_model(model_name, input_channels, 10, hidden_channels, hidden_hidden_channels,
                                   num_hidden_layers, use_intensity=False, initial=True)

    def new_make_model():
        model, regularise = make_model()
        model.linear[-1].weight.register_hook(lambda grad: 100 * grad)
        model.linear[-1].bias.register_hook(lambda grad: 100 * grad)
        return model, regularise

    name = None if dry_run else 'speech_commands'
    num_classes = 10
    return common.main(name, model_name, times, train_dataloader, val_dataloader, test_dataloader, device, new_make_model,
                       num_classes, max_epochs, lr, kwargs, step_mode=True)


def run_all(device, model_names=['staticsde', 'naivesde', 'neurallsde', 'neurallnsde', 'neuralgsde']):
    for num_layer in [1, 2, 3, 4]:
        for hidden in [16, 32, 64, 128]:
            model_kwargs = dict(staticsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                                naivesde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                                neurallsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                                neurallnsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                                neuralgsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),)
            for model_name in model_names:
                main(device, model_name=model_name, **model_kwargs[model_name])

for _ in range(5):
    run_all('cuda')