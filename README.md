# fastai_helpers -- Fastai utility functions

### Key Files

#### stats.py
- StatsRepo()
  - .add(), .show_results(), .restore(), .save()

#### train.py
- _get_learner()
  - higher level function to create learner, inclusing cupping of pre-trained model
  - model_cutter():  used for custom layer clipping of pre-trained model
- lr_find()
  - lrfinder + plots
  - Learner.Recorder.plot2(): returns smoothed plot with suggestions
- train()
  - supports both one-cycle and flat annealing
- Learner.svalidate()
  - silently validate -- compute validation results without updating Jupyter cell output 
- get_best_stats()
  - Returns best result from recorder history
- get_val_stats()
  - Returns final result from recorder history
- Thresholded metrics for multi-label classification: 
  - error_02(), accuracy_02(), error_05(), accuracy_05()
- mishify()
  - modifies resnet and xresnet models to use MISH activation function in place of RELU.
  - note: Includes version of MISH with application (mish source: https://github.com/lessw2020/mish)


#### other_helpers.py
- verify_gpu()
  - checks GPU status
- reset_seeds()
  - initialized all random seeds to known value

### Files requiring reorganization

#### helpers.py
- Interpreter:
  - _get_interp()
  - analyze_interp()
  - compute_acc()
  - combine_predictions()


#### fastai_addons.py
- plot_confusion_matrix()
  - plots confusion matrix with clean-up of distplay boundaries
- plot_confusion_matrix_thresh()
  - plots confusion matrix after thresholding values
  - threshold_confusion_matrix(): computes confusion matric using threshold
- interpretation_summary()
  - Outputs various stats (summary, cm, threshold analysis)
- get_accuracy()
- analyze_confidence()
- accuracy_vs_threshold()
- show_incremental_accuracy()
- analyze_low_confidence()