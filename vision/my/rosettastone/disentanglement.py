import copy

import numpy as np
from tqdm import trange
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
    
from rosettastone.partial_freezing import freeze_conv2d_params, freeze_linear_params


def freezer(module, requires_grad=False):
    for param in module.parameters():
        param.requires_grad = requires_grad


def weight_reset(m):
    reset_parameters = getattr(m, 'reset_parameters', None)
    if callable(reset_parameters):
        m.reset_parameters()



def get_named_parameters(self):
    return [name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad]



def train(dnet,
          train_dataloader,
          first_concept_dataloader, second_concept_dataloader,
          epochs, gamma,
          device):
    
    dnet.eval()
    
    optimizer = optim.Adam(dnet.parameters(), lr=.001)
    example_ct = 0  # number of examples seen
    batch_ct = 0
        
    for epoch in trange(epochs):
        # TODO: is it the right way?
        for indff_data, cnpt1_data, cnpt2_data in zip(train_dataloader,
                                                      first_concept_dataloader,
                                                      second_concept_dataloader):
            
            x_indff, _ = indff_data
            x_indff = x_indff.to(device)
            
            x_cnpt1, _ = cnpt1_data
            x_cnpt1 = x_cnpt1.to(device)

            x_cnpt2, _ = cnpt2_data
            x_cnpt2 = x_cnpt2.to(device)

            (indff_loss,
             spc11_loss, spc22_loss,
             spc12_loss, spc21_loss) = dnet.generate_losses(x_indff, x_cnpt1, x_cnpt2)

            indff_loss = indff_loss.mean()
            
            loss = indff_loss - gamma/2 * (spc11_loss + spc22_loss - spc12_loss - spc21_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            
            example_ct +=  len(x_indff)
            batch_ct += 1

                        
            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                                

                loss_val = float(loss)
                wandb.log({'epoch': epoch, 'loss': loss_val,
                           'indff_loss': float(indff_loss),
                           'spc11_loss': float(spc11_loss),
                           'spc22_loss': float(spc22_loss),
                           'spc12_loss': float(spc12_loss),
                           'spc21_loss': float(spc21_loss),
                          },
                          step=example_ct)
                print(f'Loss after ' + str(example_ct).zfill(5) + f' examples: {loss_val:.3f}')

    return dnet


def evaluate(dnet, test_dataloader, device):
    accs = {}
    
    dnet.eval()

    with torch.no_grad():
        for branch in ('orig', 'splitted'):
            model = getattr(dnet, f'{branch}_net')
            correct = 0
            total = 0
            for data in test_dataloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            accs[f'test_accuracy_{branch}'] = correct / total

    wandb.log(accs)
    #torch.onnx.export(dnet.splitted_net, images, 'model.onnx')
    #wandb.save('dnet_release.onnx')

    return accs


def disentangle_cnn(net,
                        disentanglement_module_name,
                        disentanglement_neuron_index,
                        indifference_module_name,
                        train_dataloader,
                        first_concept_dataloader, second_concept_dataloader,
                        test_dataloader,
                        epochs=3, gamma=10e-5,
                        with_wandb=True,
                        device='cpu'):

    config = {'disentanglement_module_name': disentanglement_module_name,
              'disentanglement_neuron_index': disentanglement_neuron_index,
              'indifference_module_name': indifference_module_name,
              'epochs': epochs,
              'gamma': gamma}

    with wandb.init(project='disentanglement', config=config,
                    save_code=True,
                    mode='online' if with_wandb else 'disabled'):
        
        dnet = DisentangleNet(net,
                              disentanglement_module_name,
                              disentanglement_neuron_index,
                              indifference_module_name,
                              device=device)

        
        wandb.watch(dnet, log='all', log_freq=100)

        dnet = train(dnet,
              train_dataloader, test_dataloader,
              first_concept_dataloader, second_concept_dataloader,
                     epochs, gamma,
              device=device)

        accs = evaluate(dnet, test_dataloader,
                              device=device)
    
    return dnet, accs


def splitting_operation(orig_layer, neuron, device):

    orig_layer_size = orig_layer.out_channels
    alt_splitted_layer_size = orig_layer_size + 1

    with torch.no_grad():
        splitted_layer = nn.Conv2d(orig_layer.in_channels,
                                       alt_splitted_layer_size,
                                       orig_layer.kernel_size,
                                       orig_layer.stride, orig_layer.padding,
                                       orig_layer.dilation, orig_layer.groups,
                                       orig_layer.bias is not None, orig_layer.padding_mode)

        splitted_layer.weight[:orig_layer.out_channels, :, :, :] = copy.deepcopy(orig_layer.weight)
        new_neuron_weight = copy.deepcopy(orig_layer.weight[neuron, :, :, :])
        new_neuron_weight_noise = torch.normal(0, 0.1, orig_layer.weight.shape[1:])  # to break symmetry
        new_neuron_weight_noise = new_neuron_weight_noise.to(new_neuron_weight)
        splitted_layer.weight[-1, :, :, :] = (new_neuron_weight
                                              + new_neuron_weight_noise)  

        if orig_layer.bias is not None:
            splitted_layer.bias[:orig_layer.out_channels] = copy.deepcopy(orig_layer.bias)
            splitted_layer.bias[-1] = copy.deepcopy(orig_layer.bias[neuron])

        assert splitted_layer.weight.shape[:2] == (alt_splitted_layer_size, orig_layer.in_channels)

    splitted_layer.to(device)
    freezer(splitted_layer, True)

    not_splitted_neurons = list(range(alt_splitted_layer_size))
    not_splitted_neurons.remove(neuron)
    not_splitted_neurons.remove(alt_splitted_layer_size-1)
    
    neuron_freezing_hooks = freeze_conv2d_params(splitted_layer,
                                                  not_splitted_neurons,
                                                )#bias_indices=[])

    return splitted_layer, neuron_freezing_hooks


def succeeding_operation(orig_layer, neuron, orig_splitted_layer_size, device):

    assert isinstance(orig_layer, nn.Linear)

    in_feature_map_size = int(orig_layer.in_features
                              / orig_splitted_layer_size)
    alt_splitted_layer_size = orig_splitted_layer_size + 1
    alt_succeeding_in_features_size = (in_feature_map_size
                                       * alt_splitted_layer_size)

    with torch.no_grad():
        succeeding_splitted_layer = nn.Linear(alt_succeeding_in_features_size,
                                                                    orig_layer.out_features,
                                                                    orig_layer.bias is not None)

        succeeding_splitted_layer.weight[:, :orig_layer.in_features] = copy.deepcopy(orig_layer.weight)
        succeeding_splitted_layer.bias = copy.deepcopy(orig_layer.bias)

        fc_input_splitted_neuron_slice = slice(neuron*in_feature_map_size, ((neuron+1)*in_feature_map_size))
        fc_input_splitted_neuron_weight = copy.deepcopy(orig_layer.weight[:, fc_input_splitted_neuron_slice])
        succeeding_splitted_layer.weight[:, -in_feature_map_size:] = copy.deepcopy(orig_layer.weight[:, fc_input_splitted_neuron_slice])

    succeeding_splitted_layer.to(device)
    freezer(succeeding_splitted_layer, True)
    
    not_splitted_neurons = list(range(alt_splitted_layer_size))
    not_splitted_neurons.remove(neuron)
    not_splitted_neurons.remove(alt_splitted_layer_size-1)
    fc_grouped_neurons = [np.arange(ind*in_feature_map_size,
                                   (ind+1)*in_feature_map_size)
                                      for ind in not_splitted_neurons]
    fc_expanded_neuron = np.concatenate(fc_grouped_neurons)
    
    
    neuron_freezing_hooks = freeze_linear_params(succeeding_splitted_layer,
                                               fc_expanded_neuron,
                                               direction='input',
                                                )#bias_indices=[])
    
    return succeeding_splitted_layer, neuron_freezing_hooks