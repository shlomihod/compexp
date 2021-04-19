import random
from collections import defaultdict
from functools import partial

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from tqdm import trange

import torch
import torchvision
from torch.nn.modules.upsampling import Upsample

from lucent.optvis import render#, param, transform, objectives


UPSAMPLE = Upsample(size=(224, 224), mode='bilinear', align_corners=False)


def crop_by_max_activation(raw_acts, img, quantile_threshold=0.1, shape=(224, 224)):
    upsampled_acts = UPSAMPLE(raw_acts[None, None, :, :]).cpu().numpy()#upsample(raw_acts.cpu().numpy(), shape)
    mask_threshold = np.quantile(upsampled_acts, 1 - quantile_threshold)
    mask = upsampled_acts > mask_threshold
    if not mask.any():
        return None

    _, _, rows, cols = np.nonzero(mask)
    min_ = min(rows.min(), cols.min())
    max_ = max(rows.max(), cols.max())
    cropped_img = img[None, :, min_:max_, min_:max_]
    return UPSAMPLE(cropped_img)


def tensor2img(img):
    npimg = img.numpy()
    return np.transpose(npimg, (1,2,0))


def show(img):
    plt.imshow(tensor2img(img), interpolation='nearest')


def show_grid(imgs, nrow=10, padding=5, figsize=None):
    grid = torchvision.utils.make_grid(imgs,
                                       nrow=nrow, padding=padding,
                                       )#normalize=True)
    if figsize is not None:
        plt.subplots(1, figsize=figsize)
        
    return show(grid)

def cosine_similarity_reprs(representations):
    representations = representations.cpu()
    normalized_representations = representations / torch.norm(representations, dim=1)[:, None]
    cosine_similarity = (normalized_representations @ normalized_representations.T).detach().numpy()
    return cosine_similarity


def _inner_cosine_similarity_heatmap(representations, images, max_num=20):
    
    representations = representations[:max_num]
    images = images[:max_num]

    cosine_similarity = cosine_similarity_reprs(representations)

    mask = np.zeros_like(cosine_similarity)
    mask[np.triu_indices_from(mask)] = True

    size = 0.037
    #fs = 25 * size * len(images)
    with sns.axes_style('white'):
        f, ax = plt.subplots(figsize=(15, 15))
        
        plt.tight_layout()

        ax = sns.heatmap(cosine_similarity,# mask=mask.T,
                        square=True,
                        vmax=1, vmin=min(0, cosine_similarity.min()),
                        cmap='coolwarm')#'YlGnBu')
        ax.invert_yaxis()

        xl, yl, xh, yh = np.array(ax.get_position()).ravel()
        
        w = xh - xl
        h = yh - yl

        for index in range(len(images)):
            img = tensor2img(images[index].detach().cpu())

            xp = xl + ax.get_xticklabels()[index]._x * size * 1.03
            xax = f.add_axes([xp - size * 0.5, yl - size * 1.5, size, size])
            xax.axison = False
            imgplot = xax.imshow(img)

            yp = yl + ax.get_yticklabels()[index]._y * size * 1.03
            yax = f.add_axes([xl - size * 1.5, yp - size * 0.5, size, size])
            yax.axison = False
            imgplot = yax.imshow(img)

            
def get_conv_layers(model):
    return {name: layer for name, layer in model.named_modules()
                if isinstance(layer, torch.nn.Conv2d)}


# todo, make into a context manager!!!
def hook_activations(model, modules):


    activations = defaultdict(list)

    def save_activation(name, mod, inp, out):
        activations[name].append(out.cpu())
    
    handels = {}
    
    for name, layer in modules.items():
        handels[name] = layer.register_forward_hook(partial(save_activation, name))
        
    return activations, handles



# https://gist.github.com/Tushar-N/680633ec18f5cb4b47933da7d10902af
def harvest_activations(model, dataloader, module_names=None):
  
    modules = {name: model._modules.get(name) for name in module_names}
    
    activations, handles = hook_activations(model, modules)
    

    num_correct = 0
    num_total = 0

    for batch in dataloader:
        x, y = batch
        # TODO: device
        x = x.cuda()
        y = y.cuda()
        
        with torch.no_grad():
            logits = model(x)

        y_hat = logits.argmax(dim=1)
        #print(logits[:, 206][:10], logits.max(dim=1)[:10])
        num_correct += (y == y_hat).sum().item()
        num_total += len(y)

        #break
    print('Acc:', num_correct / num_total)

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    all_activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

    # just print out the sizes of the saved activations as a sanity check
    for k, v in all_activations.items():
        print (k, v.size(), modules[k])
    
    for handle in handels.values():
        handle.remove()

    return all_activations


def get_neuron_max_activations(activations, dataset, layer, neuron, top_k=100):
    # TOTAL_QUANTILE
    top_img_acts, top_img_indices = activations[layer].mean(dim=[2, 3])[:, neuron].topk(k=top_k)
    top_raw_all_acts = activations[layer][top_img_indices]
    top_raw_acts = top_raw_all_acts[:, neuron, :, :]
    top_reprs = top_raw_all_acts.amax(dim=[2, 3])
    xs, ys = zip(*(dataset[idx] for idx in top_img_indices))
    xs = torch.stack(xs)
    xs = (xs - xs.min()) / (xs.max() - xs.min())
    return top_img_acts, top_img_indices, top_raw_acts, top_reprs, xs, ys


def plot_neuron_max_activations(activations, dataset, layer, neuron, cropped=True,
                                top_k=100, nrows=10, figsize=(20, 20)):
    (top_img_acts,
     top_img_indice,
     top_raw_acts,
     top_reprs,
     xs, ys) = get_neuron_max_activations(activations, dataset, layer, neuron, top_k)
    
    if cropped:
        all_cropped_xs = [crop_by_max_activation(raw_acts, xs) for raw_acts, xs in zip(top_raw_acts, xs)]
        xs = torch.cat([x for x in all_cropped_xs if x is not None])

    return show_grid(xs, nrow=nrows, figsize=figsize)


def calc_neuron_metrics(activations, dataset, layer, neuron, top_k=100):
    metrics = {}

    (top_img_acts,
     top_img_indice,
     top_raw_acts,
     top_reprs,
     xs, ys) = get_neuron_max_activations(activations, dataset, layer, neuron, top_k)
        
    cs = cosine_similarity_reprs(top_reprs)
    # coherence = -diversity
    coherence = (cs.sum() - cs.trace()) / (2 * len(cs))

    metrics['coherence'] = coherence
    metrics['nun_class'] = len(set(ys))
    metrics['min'] = top_img_acts.min().item()
    metrics['max'] = top_img_acts.max().item()
    metrics['median'] = top_img_acts.median().item()

    return metrics


def cosine_similarity_heatmap(activations, dataset, layer, neuron, max_num=20):
    (top_img_acts,
     top_img_indice,
     top_raw_acts,
     top_reprs,
     xs, ys) = get_neuron_max_activations(activations, dataset, layer, neuron, max_num)

    return _inner_cosine_similarity_heatmap(top_reprs, xs)


# https://discuss.pytorch.org/t/shuffle-issue-in-dataloader-how-to-get-the-same-data-shuffle-results-with-fixed-seed-but-different-network/45357/8
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def visualize_neuron(layer, neuron, model, activations, dataset):
    plot_neuron_max_activations(activations, dataset, layer, neuron)
    plot_neuron_max_activations(activations, dataset, layer, neuron, cropped=False)
    render.render_vis(model, f'{layer}:{neuron}', show_inline=True)
    
