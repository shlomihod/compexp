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


class DisentangleNet(nn.Module):

    def __init__(self, net, disentanglement_module_name, disentanglement_neuron_index,
                 indifference_module_name, build_fn=None, device='cpu'):
        super().__init__()
        self.net = copy.deepcopy(net)
        self._was_released = False

        self.disentanglement_module_name = disentanglement_module_name
        self.disentanglement_neuron_index = disentanglement_neuron_index
        self.indifference_module_name = indifference_module_name

        # Needed for the partial freezing hooks
        self.device = device
        
        self.neuron_freezing_hooks = []
        
        self.build = self._default_build if build_fn is None else build_fn
        self.build()
            
        freezer(self.net)
        self.to(self.device)

    def _default_build(self):
        pre_module = nn.Sequential()
        orig_disentanglement_module = nn.Sequential()
        orig_post_module = nn.Sequential()
        alt_disentanglement_module = nn.Sequential()
        orig_post_module = nn.Sequential()
        alt_post_module = nn.Sequential()
        final_module = nn.Sequential()

        modules_iter = self.net.named_children()

        for name, layer in modules_iter:
            if name == self.disentanglement_module_name:
                break
            pre_module.add_module(name, copy.deepcopy(layer))
            freezer(pre_module)

        assert isinstance(layer, nn.Conv2d)

        
        orig_disentangled_layer_size = layer.out_channels
        alt_disentangled_layer_size = orig_disentangled_layer_size + 1
        
        with torch.no_grad():
            layer_copy = copy.deepcopy(layer)
            disentangled_layer = nn.Conv2d(layer.in_channels,
                                           alt_disentangled_layer_size,
                                           layer.kernel_size,
                                           layer.stride, layer.padding,
                                           layer.dilation, layer.groups,
                                           layer.bias is not None, layer.padding_mode)
            
            disentangled_layer.weight[:layer.out_channels, :, :, :] = layer_copy.weight
            disentangled_layer.bias[:layer.out_channels] = layer_copy.bias
            
            disentangled_layer.weight[-1, :, :, :] = layer_copy.weight[self.disentanglement_neuron_index, :, :, :]
            disentangled_layer.bias[-1] = layer_copy.bias[self.disentanglement_neuron_index]
            
            assert disentangled_layer.weight.shape[:2] == (alt_disentangled_layer_size, layer.in_channels)

            alt_disentanglement_module.add_module(name, disentangled_layer)
            #alt_disentanglement_module.apply(weight_reset)
            
            orig_disentanglement_module.add_module(name, copy.deepcopy(layer))
        
        freezer(orig_disentanglement_module)

        disentangled_layer.to(self.device)
        non_disentanglement_neurons = list(range(layer.out_channels))
        non_disentanglement_neurons.remove(self.disentanglement_neuron_index)
        self.neuron_freezing_hooks.append(freeze_conv2d_params(disentangled_layer,
                                                               non_disentanglement_neurons))


        is_first_post_parameterized_layer = True
        # TODO: take care of the immidate layer
        for name, layer in modules_iter:

            with torch.no_grad():
                layer_copy = copy.deepcopy(layer)
                post_disentangled_layer = layer_copy  # TODO: refactor me!
                if is_first_post_parameterized_layer:
                    if isinstance(layer, nn.Conv2d):
                        raise NotImplemented()
                        is_first_post_parameterized_layer = False

                        post_disentangled_layer = nn.Conv2d(alt_disentangled_layer_size,
                                                            layer.out_channels,
                                                            layer.kernel_size,
                                                            layer.stride, layer.padding,
                                                            layer.dilation, layer.groups,
                                                            layer.bias is not None, layer.padding_mode)
                        
                        # todo copy weight, bias up and down
                    elif isinstance(layer, nn.Linear):
                        is_first_post_parameterized_layer = False
                        
                        in_feature_map_size = int(layer.in_features
                                                  / orig_disentangled_layer_size)
                        alt_post_in_features_size = (in_feature_map_size
                                                     * alt_disentangled_layer_size)
                        
                        post_disentangled_layer = nn.Linear(alt_post_in_features_size,
                                                            layer.out_features,
                                                            layer.bias is not None)
                        
                        post_disentangled_layer.weight[:, :layer.in_features] = layer_copy.weight
                        post_disentangled_layer.bias = layer_copy.bias

                        post_disentangled_layer.weight[:, -1] = layer_copy.weight[:, self.disentanglement_neuron_index]
            
                        post_disentangled_layer.to(self.device)


                        fc_grouped_neurons = [np.arange(neuron*in_feature_map_size,
                                                       (neuron+1)*in_feature_map_size)
                                                          for neuron in non_disentanglement_neurons]
                        fc_expanded_neuron = np.concatenate(fc_grouped_neurons)
                        self.neuron_freezing_hooks.append(freeze_linear_params(post_disentangled_layer,
                                                                   fc_expanded_neuron,
                                                                   direction='input'))
                # TODO: not sure it is needed!
                else:
                    freezer(post_disentangled_layer)

                alt_post_module.add_module(name, post_disentangled_layer)
                orig_post_module.add_module(name, copy.deepcopy(layer))
                freezer(orig_post_module)                
                if not is_first_post_parameterized_layer:
                    break
        for name, layer in modules_iter:
            with torch.no_grad():
                final_module.add_module(name, copy.deepcopy(layer))
            if name == self.indifference_module_name:
                break
        freezer(final_module)


        self.add_module('pre_module', pre_module)
        self.add_module('orig_disentanglement_module', orig_disentanglement_module)
        self.add_module('alt_disentanglement_module', alt_disentanglement_module)
        self.add_module('orig_post_module', orig_post_module)
        self.add_module('alt_post_module', alt_post_module)
        self.add_module('final_module', final_module)


    def indifference_loss(self, x):
        x = self.pre_module(x)

        x_orig = self.orig_disentanglement_module(x)
        x_orig = self.orig_post_module(x_orig)
        
        x_alt = self.alt_disentanglement_module(x)
        x_alt = self.alt_post_module(x_alt)

        return (torch.flatten(F.mse_loss(x_orig, x_alt, reduction='none'), 1)
                             .mean())


    def specificity_loss(self, x, neuron_index):
        x = self.pre_module(x)
        x = self.alt_disentanglement_module(x)
        return torch.mean(x[:, neuron_index, :, :]**2)
    
        
    def forward(self, x_indff, x_cnpt1, x_cnpt2):
        return (self.indifference_loss(x_indff),
                self.specificity_loss(x_cnpt1, self.disentanglement_neuron_index),
                self.specificity_loss(x_cnpt2, -1),
                self.specificity_loss(x_cnpt2, self.disentanglement_neuron_index),
                self.specificity_loss(x_cnpt1, -1))

    
    def predict(self, x, branch='alt'):
        assert branch in ('orig', 'alt')
        
        x = self.pre_module(x)
        x = getattr(self, f'{branch}_disentanglement_module')(x)
        x = getattr(self, f'{branch}_post_module')(x)
        x = self.final_module(x)
        return x
        

    def train(self, mode=True):
        if self._was_released:
            raise ValueError('The network was released and cannot be trained.')
            
        super().train(mode)
    
    # TODO: fix me!
    def release(self):
        self._was_released = True
        for hook in self.neuron_freezing_hooks:
            hook.remove()
        
        release_net = nn.Sequential(*self.pre_module.children(),
                                    *self.alt_disentanglement_module.children(),
                                    *self.alt_post_module.children(),
                                    *self.final_module.children())
        freezer(release_net, True)
        
        return release_net


    def get_named_parameters(self):
        return [name
                for name, parameter in self.named_parameters()
                if parameter.requires_grad]


def train(dnet,
          train_dataloader, test_dataloader,
          first_concept_dataloader, second_concept_dataloader,
          epochs, gamma,
          device):
    
    dnet.train()
    
    optimizer = optim.Adam(dnet.parameters())#, lr=0.01, momentum=0.9)
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in trange(epochs):
        # TODO: is it the right way?
        for indff_data, cnpt1_data, cnpt2_data in zip(train_dataloader,
                                                      first_concept_dataloader,
                                                      second_concept_dataloader):
        #joint_dataloader:
            
            x_indff, _ = indff_data
            x_indff = x_indff.to(device)
            
            x_cnpt1, _ = cnpt1_data
            x_cnpt1 = x_cnpt1.to(device)

            x_cnpt2, _ = cnpt2_data
            x_cnpt2 = x_cnpt2.to(device)

            (indff_loss,
             spc11_loss, spc22_loss,
             spc12_loss, spc21_loss) = dnet(x_indff, x_cnpt1, x_cnpt2)
            loss = indff_loss - gamma/2 * (spc11_loss + spc22_loss - spc12_loss - spc21_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            
            example_ct +=  len(x_indff)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                loss_val = float(loss)
                # where the magic happens
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
        for branch in ('orig', 'alt'):
            correct = 0
            total = 0
            for data in test_dataloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                outputs = dnet.predict(images, branch)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            accs[f'test_accuracy_{branch}'] = correct / total

    
        print(accs)
        wandb.log(accs)

    # TODO: remove hooks before saving
    # torch.onnx.export(dnet.release(), images, 'model.onnx')
    # wandb.save('dnet_release.onnx')


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

    return splitted_layer, [] #neuron_freezing_hooks


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
        #new_neuron_weight_noise = torch.normal(0, 0.1, fc_input_splitted_neuron_weight.shape)
        #new_neuron_weight_noise = new_neuron_weight_noise.to(fc_input_splitted_neuron_weight.device)
        succeeding_splitted_layer.weight[:, -in_feature_map_size:] = copy.deepcopy(orig_layer.weight[:, fc_input_splitted_neuron_slice])
        #(fc_input_splitted_neuron_weight
                                                                     # + new_neuron_weight_noise)
                                                                      
                                                                      

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
    
    return succeeding_splitted_layer, []#neuron_freezing_hooks
