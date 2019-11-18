# fastai_helpers
Fastai utility functions

Includes:
- fastai_helpers.py:
  - reset_seeds()
  - plot_confusion_matrix()

- fastai_addons.py:
  - model_cutter():  used for custom layer clipping of pre-trained model
  - Learner.svalidate() and silent_validate(): compute validation results without updating Jupyter cell output 
  - get_best_stats(): When used with SaveModel Callback, returns best value
    - get_val_stats() returns  recorder stats for all epochs during training
  - Recorder.plot2:  Extension to Learn.Recorder.plot() to returned smoothed plot and suggested learning rates
  - plot_confusion_matrix():  plots confusion matrix with clean-up of distplay boundaries
  - plot_confusion_matrix_thresh(): plots confusion matrix after thresholding values
    - threshold_confusion_matrix(): computes confusion matric using threshold
  - interpretation_summary():  Outputs various stats (summary, cm, threshold analysis)
  - get_accuracy()
  - analyze_confidence()
  - accuracy_vs_threshold()
  - show_incremental_accuracy()
  - analyze_low_confidence()

- helpers.py:
  - get_best_stats()
  - show_results()
  - _get_learner()
  - _do_train()
  - _get_interp()
  - analyze_interp()
  - compute_acc()
  - combine_predictions()
  - StatsRepo()
  


- mishify.py:
  - modifies resnet and xresnet models to use MISH activation function in place of RELU.
  - note: Includes version of MISH with application (mish source: https://github.com/lessw2020/mish)

- verify_gpu_torch.py:
  - checks GPU status
