from fastai.vision import *
from fastai.callbacks import *
from fastai.basic_train import Recorder
from fastai.core import ifnone, defaults, Any
import numpy as np
from fastai.torch_core import to_np
import matplotlib.pyplot as plt
from typing import Optional
import scipy
import itertools
import fastai
import torchvision



# from fastai.vision import Learner
# from fastai.callbacks import SaveModelCallback


# wrap with partial
def _get_learner(db=None, model=None, model_dir=None, unfreeze=False, 
                 metrics=[error_rate, accuracy], cut=None, **kwargs):
    # verity below params have been replaced using partial
    assert db is not None
    assert model is not None
    assert model_dir is not None

    if cut:
        assert isinstance(cut, int)
        mc = partial(model_cutter, select= [cut])
        my_split_on = lambda m: (m[0][cut],m[1])
        learn = cnn_learner(db, model, metrics=metrics, 
                            cut=mc, split_on=my_split_on,
                            model_dir=model_dir, **kwargs).to_fp16()
    else:
        learn = cnn_learner(db, model, metrics=metrics, 
                        model_dir=model_dir, **kwargs).to_fp16()
    return learn


def model_cutter(model, select=[]):
    cut = select[0]
    ll = list(model.children())
    if len(select) == 1:
        if cut == 0 and isinstance(ll[0], torch.nn.modules.container.Sequential):
            return ll[0]
        else:
            return nn.Sequential(*ll[:cut+1])
    else:
        if cut == 0:
            return model_cutter(ll[0], select=select[1:])
        else:
            new_ll = ll[:cut] +  [model_cutter(ll[cut], select=select[1:])]
            return nn.Sequential(*new_ll)


def lr_find(learn, use_fp16=True, **kwargs):
    if use_fp16: learn.to_fp16()
    learn.lr_find(**kwargs)
    learn.recorder.plot()
    learn.recorder.plot2()


def my_smooth(sig, w=2):
    sig_p = np.pad(sig, (w,w), 'edge')
    return np.array([np.mean(sig_p[i:i+2*w+1]) for i in range(len(sig_p)-2*w)])


def plot2(self, skip_start:int=10, skip_end:int=5, suggestion:bool=True, return_fig:bool=None, win=3,
         **kwargs)->Optional[plt.Figure]:
    "Plot learning rate and losses, trimmed between `skip_start` and `skip_end`. Optionally plot and return min gradient"
    lrs = self._split_list(self.lrs, skip_start, skip_end)
    losses = self._split_list(self.losses, skip_start, skip_end)
    losses = [x.item() for x in losses]
    all_losses = [losses]
    
    #if 'k' in kwargs: losses = self.smoothen_by_spline(lrs, losses, **kwargs)
    fig, ax = plt.subplots(1,1)
    ax.plot(lrs, losses)
    
    if win is not None: 
        losses2 = my_smooth(losses, w=win)
        all_losses.append(losses2)
        ax.plot(lrs, losses2, 'g', lw=0.5)
    
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
    if suggestion:
        for i, l in enumerate(all_losses):
            tag = '' if i == 0 else ' (smoothed)'
            try: mg = (np.gradient(np.array(l))).argmin()
            except:
                print(f"Failed to compute the gradients{tag}, there might not be enough points.")
                return
            print(f"Min numerical gradient: {lrs[mg]:.2E} {tag}")
            color = 'r' if i == 0 else 'g'
            ax.plot(lrs[mg],losses[mg],markersize=10,marker='o',color=color)
            if i == 0: 
                self.min_grad_lr = lrs[mg]
                ml = np.argmin(l)
                ax.plot(lrs[ml],losses[ml],markersize=8,marker='o',color='k')
                print(f"Min loss divided by 10: {lrs[ml]/10:.2E}")
                ax.plot([lrs[ml]/10, lrs[ml]/10], [np.min(l), np.max(l)], 'k--', alpha=0.5)
                #print(np.min(l), np.max(l))
            elif i == 1:
                self.min_grad_lr_smoothed = lrs[mg]
        
    if ifnone(return_fig, defaults.return_fig): return fig
    try:
        if not IN_NOTEBOOK: plot_sixel(fig)
    except: pass
        
Recorder.plot2 = plot2



def silent_validate(model:nn.Module, dl:DataLoader, loss_func:OptLossFunc=None, cb_handler:Optional[CallbackHandler]=None,
             average=True, n_batch:Optional[int]=None)->Iterator[Tuple[Union[Tensor,int],...]]:
    """Calculate `loss_func` of `model` on `dl` in evaluation mode.  
       Note:  This version does not overwrite results from training"""
    model.eval()
    with torch.no_grad():
        val_losses,nums = [],[]
        if cb_handler: cb_handler.set_dl(dl)
        for xb,yb in dl:
            if cb_handler: xb, yb = cb_handler.on_batch_begin(xb, yb, train=False)
            val_loss = loss_batch(model, xb, yb, loss_func, cb_handler=cb_handler)
            val_losses.append(val_loss)
            if not is_listy(yb): yb = [yb]
            nums.append(first_el(yb).shape[0])
            if cb_handler and cb_handler.on_batch_end(val_losses[-1]): break
            if n_batch and (len(nums)>=n_batch): break
        nums = np.array(nums, dtype=np.float32)
        if average: return (to_np(torch.stack(val_losses)) * nums).sum() / nums.sum()
        else:       return val_losses


def _svalidate(self, dl=None, callbacks=None, metrics=None):
        "Validate on `dl` with potential `callbacks` and `metrics`."
        dl = ifnone(dl, self.data.valid_dl)
        metrics = ifnone(metrics, self.metrics)
        cb_handler = CallbackHandler(self.callbacks + ifnone(callbacks, []), metrics)
        cb_handler.on_train_begin(1, None, metrics); cb_handler.on_epoch_begin()
        val_metrics = silent_validate(self.model, dl, self.loss_func, cb_handler)
        cb_handler.on_epoch_end(val_metrics)
        return cb_handler.state_dict['last_metrics']
    
Learner.svalidate = _svalidate





# wrap with partial
def _train(learn:Learner, n_epochs:int, lr:float=4e-3, start_pct:float=0.72, 
          key=None, mixup=None, monitor_metric='error', stats_repo=None, use_slice=False,
          use_fp16=True, show_plots=True, save_model=True, use_fa=False, unfreeze=None,
          **kwargs):

    if key: key = f'{key}_c{n_epochs}_lr{lr}_'
    if use_fp16: learn.to_fp16()

    
    if isinstance(unfreeze, int):
        if key: key = f'{key}_uf{unfreeze}' 
        learn.freeze_to(unfreeze)
    elif unfreeze:
        if key: key = f'{key}_ufa'
        learn.freeze()

    if mixup:
        if isinstance(mixup, float):
            if key: key = f'{key}_mu{mixup}'
            learn.mixup(mixup)
        else:
            if key: key = f'{key}_mu'
            learn.mixup()

    if use_slice:
        lr = slice(lr)
        if key: key = f'{key}_sl'

    if key:
        if use_fa:
            key = f'{key}_fa'
        else:
            key = f'{key}_oc'

    if key: print(key)

    if key and stats_repo and stats_repo.exists(key):
        print(f'skipping training, result already exists: {key}')
        return
            
    save_cb = SaveModelCallback(learn, every='improvement', monitor=monitor_metric, name='best')
    if use_fa:
        sched = get_flat_anneal(learn, lr, n_epochs, start_pct)
        learn.fit(n_epochs, callbacks=[sched, save_cb], **kwargs)
    else:
        learn.fit_one_cycle(n_epochs, callbacks=[save_cb], **kwargs)

    if show_plots:
        learn.recorder.plot_losses()
        plt.show()
        learn.recorder.plot_metrics()
        plt.show()
    
    if key and stats_repo:
        stats =  get_best_stats(learn, monitor_metric) 
        stats_repo.add([key, stats])
        print('updated stats')

    if save_model and key:  
        learn.save(key)


def get_flat_anneal(learn:Learner, lr:float, n_epochs:int, start_pct:float):
    n = len(learn.data.train_dl)
    anneal_start = int(n*n_epochs*start_pct)
    anneal_end = int(n*n_epochs) - anneal_start
    phases = [TrainingPhase(anneal_start).schedule_hp('lr', lr),
              TrainingPhase(anneal_end).schedule_hp('lr', lr, anneal=annealing_cos)]
    sched = GeneralScheduler(learn, phases)
    return sched


def get_best_stats(learner, monitor_metric):
    rec = learner.recorder
    keys = ['loss'] + rec.metrics_names
    results = []
    for i, loss in enumerate(rec.val_losses):
        entry = [loss] + [float(v) for v in rec.metrics[i]]
        results.append(dict(zip(keys, entry)))
    return sorted(results, key=lambda x:x[monitor_metric])[0]


def get_val_stats(learner):
    """returns final stats from recorder"""
    rec = learner.recorder
    ret = {'loss':float(rec.val_losses[-1])}
    for i, name in enumerate(rec.metrics_names):
        ret[name] = float(rec.metrics[-1][i])
    return ret

def error_02(*args, **kwargs):
    return 1.0 -accuracy_thresh(*args, thresh=0.2, **kwargs)
def accuracy_02(*args, **kwargs):
    return accuracy_thresh(*args, thresh=0.2, **kwargs)
def error_05(*args, **kwargs):
    return 1.0 -accuracy_thresh(*args, thresh=0.5, **kwargs)
def accuracy_05(*args, **kwargs):
    return accuracy_thresh(*args, thresh=0.5, **kwargs)



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
    
