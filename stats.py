from fastai.vision import *
from fastai.callbacks import SaveModelCallback
from fastai.metrics import error_rate
import os
import pickle
import scipy
from scipy.stats import gmean, hmean




class StatsRepo:
    def __init__(self, prefix, force_init=False, stats_fn=None, checkpoint=False, verbose=False):
        self.prefix = prefix
        self.checkpoint = checkpoint
        self.verbose = verbose
        self.stats_fn = Path(stats_fn) if stats_fn else Path('stats')/f'{self.prefix}_stats.p'
        if not force_init and self.stats_fn.exists():
            self.restore()
        else:
            self.clear()
            
    def add(self, val):
        self.all_results.append(val)
        if self.checkpoint:
            self.save()
            
    def clear(self):
        if self.verbose: print('initialializing stats for', self.prefix)
        self.all_results = []
        if self.checkpoint:
            self.save()
            
    def get(self):
        return self.all_results
        
    def save(self):
        with open(self.stats_fn, 'wb') as f:
            pickle.dump(self.all_results, f)
        if self.verbose: print('saved stats to:', self.stats_fn)
        
    def restore(self):
        with open(self.stats_fn, 'rb') as f:
            self.all_results = pickle.load(f)
        if self.verbose: print('restored stats from:', self.stats_fn)

    def exists(self, tag):
        return tag in [x for x, _ in self.all_results]


    def show_results(self, key=None, show_details=True, limit=None, metric='error_rate'):

        results = [x for x in self.all_results if key in x[0]] if key else self.all_results
            
        if len(results) > 1:
            loss = [stats['loss'] for key, stats in results]
            vals = [stats[metric] for key, stats in results]
            best = np.max(vals) if 'acc' in metric else np.min(vals)
            title = 'Overall' if key is None else key
            print(f'{metric} -- best: {best:.3f}  med: {np.median(vals):.3f}   loss -- best: {np.min(loss):.3f}  med: {np.median(loss):.3f} -- {key}')
            if not show_details: return
            print('')

        results = sorted(results, key=lambda x:x[1][metric], reverse='acc' in metric)
        if limit is None: limit = len(results)
        for key, stats in results[:limit]:
            print(f"{metric}: {stats[metric]:.3f}   loss:{stats['loss']:.4f} -- {key}")
   

    

def stats_repo_unit_test(prefix='unit_test'):
    stats = StatsRepo(prefix, force_init=True, stats_fn=None, checkpoint=False, verbose=True)
    print('** expected *', 'initialializing stats')

    print()
    print(stats.stats_fn)
    print('** expected *', 'stats/18_448_stats.p')

    stats.add('foobar')

    print()
    stats.save()
    print('** expected *', 'saved stats to: stats\18_448_stats.p')

    print()
    stats.restore()
    print('** expected *', 'saved stats to: stats\18_448_stats.p')

    print()
    print(stats.get())
    print('** expected *', "['foobar']")

    print()
    stats = StatsRepo(prefix, force_init=False, stats_fn=None, checkpoint=True, verbose=True)
    print('** expected *', 'restored stats from: stats\18_448_stats.p')

    print()
    stats.add('bar')
    print('** expected *', 'saved stats to: stats\18_448_stats.p')

    print()
    print(stats.get())
    print('** expected *', "['foobar', 'bar']")

    stats = StatsRepo(prefix, force_init=True, stats_fn=None, checkpoint=True, verbose=True)

    print()
    print(stats.get())
    print('** expected *', '[]') 
   