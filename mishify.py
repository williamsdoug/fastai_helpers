# Converts RESNET and XRESNET models to use mish (relaces all relu)

import fastai
import torchvision

#-----------------------------------------------------------------
# Mish code from github source below.  Included for convenience to reduce library dependency
# Notes:
#    - Code subject to Apache 2 license
#    - Mish function can be replaced using mish_fn keyword
#

#from mish import Mish       # note: mish.py missing 'import torch'
#from mxresnet import Mish   # removed to avoid dependency
import torch.nn as nn
import torch.nn.functional as F  #(uncomment if needed,but you likely already have it)
import torch

#Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
#https://arxiv.org/abs/1908.08681v1
#implemented for PyTorch / FastAI by lessw2020 
#github: https://github.com/lessw2020/mish

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x *( torch.tanh(F.softplus(x)))
        return x
    
#---------------------------------------------------------------

def _mishify(mod, mish_fn=Mish, indices=[], verbose=False, debug=False, max_layers=20):
    if debug: print('called mishify', indices)
        
    if hasattr(mod, 'relu'):
        type_suffix = str(type(mod)).split('.')[-1][:-1]
        if verbose: print('Replacing RELU with Mish:', indices+[type_suffix, 'relu'])
        mod.relu = mish_fn()
    
    # torchvision ResNet specific code
    if isinstance(mod, torchvision.models.resnet.ResNet):
        if debug: print('Found', type(mod))   
        for i in range(max_layers):
            lname = f'layer{i}'
            if hasattr(mod, lname):
                if debug: print('found', indices+['ResNet', lname])
                _mishify(getattr(mod, lname), mish_fn, indices+['ResNet', lname], verbose, debug, max_layers)
           
    # fastai XResNet specific
    elif isinstance(mod, fastai.vision.models.xresnet.ResBlock):
        if debug: print(indices, 'Found ResBlock  convs:', hasattr(mod, 'convs'))
        if hasattr(mod, 'convs'):
            _mishify(mod.convs, mish_fn, indices+['ResBlock', 'convs'], verbose, debug, max_layers)
        if debug: print('')
            
    else:
        try:
            for i in range(len(mod)):
                new_indices = indices + [i]
                if isinstance(mod[i], torch.nn.modules.activation.ReLU):
                    if verbose: print('Replacing RELU with Mish:', new_indices)
                    mod[i] = mish_fn()
                else:
                    _mishify(mod[i], mish_fn, new_indices, verbose, debug, max_layers)
        except:
            if debug: 
                print('Exception:', indices, type(mod))
                print(mod)
                print()

def mishify(model, mish_fn=Mish, **kwargs):
    """Replaces RELU with Mish activation for resnet and xresnet models.  
       mish_fn param specified class used to replace relu"""
    _mishify(model, mish_fn=mish_fn, **kwargs)
    model.cuda()
    