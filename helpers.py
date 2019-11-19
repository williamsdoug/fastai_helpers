from fastai.vision import *
from fastai.callbacks import SaveModelCallback
from fastai.metrics import error_rate
import os
import pickle
import scipy
from scipy.stats import gmean, hmean




def _get_interp(name, use_tta=False, get_learner=None):
    assert get_learner is not None
    learn = get_learner()
    learn.load(name)
    interp = learn.to_fp32().interpret(tta=use_tta)
    return interp

