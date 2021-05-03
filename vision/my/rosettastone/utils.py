import numpy as np
import matplotlib.pylab as plt
from torch.utils.data import DataLoader
import torchvision


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

def tensor2img(img):
    npimg = img.numpy()
    return np.transpose(npimg, (1,2,0))


def show(img):
    return plt.imshow(tensor2img(img), interpolation='nearest')


def show_grid(imgs, nrow=10, padding=5, figsize=None):
    grid = torchvision.utils.make_grid(imgs,
                                       nrow=nrow, padding=padding,
                                       )#normalize=True)
    if figsize is not None:
        plt.subplots(1, figsize=figsize)
        
    return show(grid)


def forward_up_to(module, last_index):
    def fn(x):
        for i in range(last_index+1):
            x = module[i](x)
        return x
    return fn


# https://gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958    
class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch
