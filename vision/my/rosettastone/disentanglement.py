import copy

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from tqdm import trange
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rosettastone.partial_freezing import freeze_conv2d_params, freeze_linear_params
from rosettastone.utils import freezer, forward_up_to


def plot_concept_activation_line(forward_fn, concept_probs, neuron, device):
    _, ax = plt.subplots(1)
    concepts = sorted(concept_probs.keys())
    probs = {c: next(iter(concept_probs[c]))[0] for c in concepts}
    y_concepts = np.array(sum([[c] * len(probs[c]) for c in concepts], []))
    imgs = torch.cat([probs[c] for c in concepts])
    imgs = imgs.to(device)
    acts = forward_fn(imgs)[:, neuron, :, :].mean(dim=(1,2)).detach().cpu().numpy()
    sort_indices = np.argsort(acts)[::-1]
    acts = acts[sort_indices]
    y_concepts = y_concepts[sort_indices]
    sns.scatterplot(x=range(len(acts)), y=acts, hue=y_concepts, ax=ax)
    ax.set_ylabel('Mean Activation')
    ax.set_xlabel('Ordered Images')
    return ax


def train(dnet,
          train_dataloader,
          first_concept, second_concept,
          concept_dataloaders,
          epochs, lr, alpha, beta,
          device,
          verbose=True):
    
    first_concept_dataloader = concept_dataloaders[first_concept]
    second_concept_dataloader = concept_dataloaders[second_concept]

    try:
        if len(alpha) == 4:
            alpha = np.array(alpha)
        else:
            raise ValueError('alpha should be number or a sequence with length 4.')
    except TypeError:
        alpha = np.array([alpha] * 4)
        
    try:
        if len(beta) == 2:
            beta = np.array(beta)
        else:
            raise ValueError('beta should be number or a sequence with length 4.')
    except TypeError:
        beta = np.array([beta] * 2)
    
    dnet.eval()

    optimizer = optim.Adam(dnet.parameters(), lr=lr)
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
             spc12_loss, spc21_loss,
             wd1_loss, wd2_loss) = dnet.generate_losses(x_indff, x_cnpt1, x_cnpt2)

            indff_loss = indff_loss.mean()
            spc11_loss = spc11_loss.mean()
            spc22_loss = spc22_loss.mean()
            spc12_loss = spc12_loss.mean()
            spc21_loss = spc21_loss.mean()

            spc_loss = (alpha[0] * spc11_loss
                        + alpha[1] * spc22_loss
                        - alpha[2] * spc12_loss
                        - alpha[3] * spc21_loss) 
            
            wd_loss = (beta[0] * wd1_loss + beta[1] * wd2_loss)

            loss = indff_loss - spc_loss + wd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            
            example_ct +=  len(x_indff)
            batch_ct += 1
                        
            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                                

                loss_val = float(loss)
                wandb.log({'epoch': epoch,
                           'loss': loss_val,
                           'indff_loss': float(indff_loss),
                           'spc_loss': float(spc_loss),
                           'wd_loss': float(wd_loss),
                           'spc11_loss': float(spc11_loss),
                           'spc22_loss': float(spc22_loss),
                           'spc12_loss': float(spc12_loss),
                           'spc21_loss': float(spc21_loss),
                           'wd1_loss': float(wd1_loss),
                           'wd2_loss': float(wd2_loss),
                          },
                          step=example_ct)
                if verbose:
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


def splitting_operation(orig_layer, neuron, new_neuron_noise_std, device):

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
        new_neuron_weight_noise = torch.normal(0, new_neuron_noise_std, orig_layer.weight.shape[1:])  # to break symmetry
        new_neuron_weight_noise = new_neuron_weight_noise.to(new_neuron_weight)
        splitted_layer.weight[-1, :, :, :] = (new_neuron_weight
                                              + new_neuron_weight_noise
                                             )  

        if orig_layer.bias is not None:
            splitted_layer.bias[:orig_layer.out_channels] = copy.deepcopy(orig_layer.bias)
            splitted_layer.bias[-1] = copy.deepcopy(orig_layer.bias[neuron])

        assert splitted_layer.weight.shape[:2] == (alt_splitted_layer_size, orig_layer.in_channels)

    splitted_layer.to(device)
    freezer(splitted_layer, True)

    not_splitted_neurons = list(range(alt_splitted_layer_size))
    bias_indices = not_splitted_neurons[:]
    not_splitted_neurons.remove(neuron)
    not_splitted_neurons.remove(alt_splitted_layer_size-1)
    
    neuron_freezing_hooks = freeze_conv2d_params(splitted_layer,
                                                  not_splitted_neurons,
                                                bias_indices=bias_indices)

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
                                               direction='input')
    
    return succeeding_splitted_layer, neuron_freezing_hooks


def plot_maxact(dnet, dataset, dataloader, neuron, which, top_k):
    net = dnet.orig_net if which == 'orig' else dnet.splitted_net
    with torch.no_grad(): 
        activations = maxact.harvest_activations(net, dataloader, {'layer': dnet.layer_getter})
        print('SSS', activations['layer'].shape)
        if which == 'orig':
            ax1 = maxact.plot_neuron_max_activations(activations, dataset, 'layer', neuron, cropped=False, top_k=top_k)
            ax2 = None
        else:
            ax1 = maxact.plot_neuron_max_activations(activations, dataset, 'layer', neuron, cropped=False, top_k=top_k)
            ax2 = maxact.plot_neuron_max_activations(activations, dataset, 'layer', -1, cropped=False, top_k=top_k)
        del activations
    return ax1, ax2


# TODO: refactor
def disentanglenet(disentanglenet_cls, model, neuron,
                    dataset, train_dataloader, test_dataloader,
                    first_concept, second_concept, concept_probs, concept_dataloaders,
                    epochs, lr, alpha, beta, new_neuron_noise_std,
                    device,
                    top_k=50,
                    with_wandb=True, with_maxact=False, with_prob=True,
                    project='disentanglement',
                    verbose=True):
    
    with wandb.init(project=project,
                    save_code=True,
                    mode='online' if with_wandb else 'disabled',
                    config={'neuron': neuron,
                            'first_concept': first_concept,
                            'second_concept': second_concept,
                            'top_k': top_k,
                            'epochs': epochs,
                            'lr': lr,
                            'alpha': alpha,
                            'beta': beta,
                            'new_neuron_noise_std': new_neuron_noise_std,
                            'verbose': verbose}):

        config = wandb.config
        
        dnet = disentanglenet_cls(model,
                                  config.neuron,
                                  config.new_neuron_noise_std,
                                  device)

        wandb.watch(dnet, log='all', log_freq=100)

        if verbose:
            print('Pre equality report...')
            dnet.equality_report()

        if with_maxact:
            if verbose:
                print('Pre generate maxact plots...')
            ax_init_orig_maxact, _ = plot_maxact(dnet, dataset, train_dataloader,
                                                 config.neuron,
                                                 'orig', top_k=top_k)

            ax_init_splitted_maxact_first, ax_init_second_maxact_splitted = plot_maxact(dnet, dataset, train_dataloader,
                                                                                        config.neuron,
                                                                                        'splitted', top_k=top_k)
            wandb.log({'init_orig_maxact': wandb.Image(ax_init_orig_maxact),
                       'init_splitted_maxact_first': wandb.Image(ax_init_splitted_maxact_first),
                       'init_second_maxact_splitted': wandb.Image(ax_init_second_maxact_splitted)})

        if with_prob:
            if verbose:
                print('Pre probing concepts...')
            wandb.log({'init_activation_line_orig':
                       wandb.Image(plot_concept_activation_line(dnet.orig_layer_forward_fn, concept_probs,
                                                                config.neuron, device)),
                       'init_activation_line_splitted_first': 
                       wandb.Image(plot_concept_activation_line(dnet.splitted_layer_forward_fn, concept_probs,
                                                                config.neuron, device)),
                       'init_activation_line_splitted_second': 
                       wandb.Image(plot_concept_activation_line(dnet.splitted_layer_forward_fn, concept_probs,
                                                                -1, device))})

        if verbose:
            print('Training...')
        train(dnet, train_dataloader,
              config.first_concept,
              config.second_concept,
              concept_dataloaders,
              epochs=config.epochs, lr=config.lr,
              alpha=config.alpha, beta=config.beta,
              device=device,
              verbose=config.verbose)

        if with_prob:
            if verbose:
                print('Post probing concepts...')
            wandb.log({'final_activation_line_splitted_first': wandb.Image(plot_concept_activation_line(dnet.splitted_layer_forward_fn, concept_probs,
                                                                                                        config.neuron, device)),
                       'final_activation_line_splitted_second': wandb.Image(plot_concept_activation_line(dnet.splitted_layer_forward_fn, concept_probs,
                                                                                                         -1, device))})

        if verbose:
            print('Post equality report...')
            dnet.equality_report()

        if verbose:
            print('Evaluating...')
        accs = evaluate(dnet, test_dataloader, device)
        if verbose:
            print(accs)

        if with_maxact:
            if verbose:
                print('Post generate maxact plots...')
            # TODO

        if verbose:
            print('Done!')
