import os
import torch

# CUDA for PyTorch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

import common_sde as common
import datasets


class InitialValueNetwork(torch.nn.Module):
    def __init__(self, intensity, hidden_channels, model):
        super(InitialValueNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(7 if intensity else 5, 256)
        self.linear2 = torch.nn.Linear(256, hidden_channels)

        self.model = model

    def forward(self, times, coeffs, final_index, **kwargs):
        *coeffs, static = coeffs
        z0 = self.linear1(static)
        z0 = z0.relu()
        z0 = self.linear2(z0)
        return self.model(times, coeffs, final_index, z0=z0, **kwargs)


def main(intensity,                                                               # Whether to include intensity or not
         device='cuda', max_epochs=200, pos_weight=10, *,                         # training parameters
         model_name, hidden_channels, hidden_hidden_channels, num_hidden_layers,  # model parameters
         dry_run=False,
         **kwargs):                                                               # kwargs passed on to cdeint

    batch_size = 1024
    lr = 1e-3

    static_intensity = intensity
    # these models use the intensity for their evolution. They won't explicitly use it as an input unless we include it
    # via the use_intensity parameter, though.
    time_intensity = intensity or (model_name in ('odernn', 'dt', 'decay'))

    times, train_dataloader, val_dataloader, test_dataloader = datasets.sepsis.get_data(static_intensity,
                                                                                        time_intensity,
                                                                                        batch_size)

    input_channels = 1 + (1 + time_intensity) * 34
    make_model = common.make_model(model_name, input_channels, 1, hidden_channels,
                                   hidden_hidden_channels, num_hidden_layers, use_intensity=intensity, initial=False)

    def new_make_model():
        model, regularise = make_model()
        model.linear[-1].weight.register_hook(lambda grad: 100 * grad)
        model.linear[-1].bias.register_hook(lambda grad: 100 * grad)
        return InitialValueNetwork(intensity, hidden_channels, model), regularise

    if dry_run:
        name = None
    else:
        intensity_str = '_intensity' if intensity else '_nointensity'
        name = 'sepsis' + intensity_str
    num_classes = 2
    return common.main(name, model_name, times, train_dataloader, val_dataloader, test_dataloader, device,
                       new_make_model, num_classes, max_epochs, lr, kwargs, pos_weight=torch.tensor(pos_weight),
                       step_mode=True)


def run_all(intensity, device, model_names=['staticsde', 'naivesde', 'neurallsde', 'neurallnsde', 'neuralgsde']):
    for num_layer in [1, 2, 3, 4]:
        for hidden in [16, 32, 64, 128]:
            model_kwargs = dict(staticsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                                naivesde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                                neurallsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                                neurallnsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                                neuralgsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),)
            for model_name in model_names:
                main(intensity, device, model_name=model_name, **model_kwargs[model_name])

for _ in range(5):
    run_all(False, 'cuda')
    run_all(True, 'cuda')
