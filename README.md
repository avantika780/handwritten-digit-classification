# Handwritten Digit Classification on MNIST using a CNN with Batch Normalisation, Squeeze-and-Excitation Attention and Augmentation

## 1. Student Details
- **Name:** Avantika Katyani
- **Registration Number:** 2430010385
- **Section:** D

## 2. Dataset
- **Name:** MNIST handwritten digits
- **Source:** `tf.keras.datasets.mnist` (LeCun et al., http://yann.lecun.com/exdb/mnist/)
- **Samples:** 70,000 (28×28 grayscale) — **Classes:** 10 (digits 0–9)
- **Split:** 54,000 train / 6,000 validation / 10,000 test (validation is a stratified 10% of the official training set, `random_state=42`)
- **Preprocessing:** scale pixels to [0,1], reshape to 28×28×1; augmentation (proposed model, training data only)

**Class distribution / imbalance**

<!--BEGIN:dataset_class-->
| Class | Train | Validation | Test | Total | % of dataset |
|---|---|---|---|---|---|
| 0.0 | 5331.0 | 592.0 | 980.0 | 6903.0 | 9.86 |
| 1.0 | 6068.0 | 674.0 | 1135.0 | 7877.0 | 11.25 |
| 2.0 | 5362.0 | 596.0 | 1032.0 | 6990.0 | 9.99 |
| 3.0 | 5518.0 | 613.0 | 1010.0 | 7141.0 | 10.2 |
| 4.0 | 5258.0 | 584.0 | 982.0 | 6824.0 | 9.75 |
| 5.0 | 4879.0 | 542.0 | 892.0 | 6313.0 | 9.02 |
| 6.0 | 5326.0 | 592.0 | 958.0 | 6876.0 | 9.82 |
| 7.0 | 5638.0 | 627.0 | 1028.0 | 7293.0 | 10.42 |
| 8.0 | 5266.0 | 585.0 | 974.0 | 6825.0 | 9.75 |
| 9.0 | 5354.0 | 595.0 | 1009.0 | 6958.0 | 9.94 |
<!--END:dataset_class-->

<!--BEGIN:dataset_imbalance-->
| Largest class | Largest count | Smallest class | Smallest count | Imbalance ratio (max/min) |
|---|---|---|---|---|
| 1.0 | 7877.0 | 5.0 | 6313.0 | 1.248 |
<!--END:dataset_imbalance-->

**Train / validation / test distribution**

<!--BEGIN:dataset_split-->
| Split | Samples | Percentage |
|---|---|---|
| Train | 54000 | 77.14 |
| Validation | 6000 | 8.57 |
| Test | 10000 | 14.29 |
| Total | 70000 | 100.0 |
<!--END:dataset_split-->

**Preprocessing steps and parameters**

<!--BEGIN:dataset_preprocessing-->
| Step | Parameters |
|---|---|
| Train/validation split | 10% of the 60,000 training images, stratified, random_state=42 |
| Pixel scaling | uint8 [0,255] -> float32 [0,1] (divide by 255) |
| Reshape | (N, 28, 28) -> (N, 28, 28, 1) |
| Augmentation (proposed model, training data only) | rotation +/-10 deg, translation +/-10%, zoom +/-10%, black fill |
<!--END:dataset_preprocessing-->

Figure: `figures/class_distribution.png`

## 3. Model
- **Baseline:** 2-block CNN (Conv32 → MaxPool → Conv64 → MaxPool → Dense128 (L2) → Dropout → Softmax), 225,034 parameters. This is the model developed in `notebooks/experiments.ipynb` (99.07% test accuracy).
- **Proposed model:** same backbone with three additions — (A) Batch Normalisation after each convolution, (B) Squeeze-and-Excitation (SE) channel attention after each convolution block, (C) on-the-fly geometric data augmentation during training. 226,710 parameters (+0.7% over baseline).
- **Pretrained model:** none — trained from scratch.
- Full layer summary: `models/model_description.txt`.

## 4. Hyperparameters

<!--BEGIN:hyperparameters-->
| Hyperparameter | Value |
|---|---|
| Learning rate | 0.001 |
| Batch size | 64 |
| Number of epochs | max 30 (early stopping, patience 5; ran 30 for proposed) |
| Optimizer | Adam |
| Loss function | Sparse categorical cross-entropy |
| Scheduler | ReduceLROnPlateau (monitor=val_loss, factor=0.5, patience=2, min_lr=1e-5) |
| Dropout | 0.3 |
| Weight decay | L2 = 0.001 on the Dense(128) kernel |
| Other parameters | Conv filters 32/64 (3x3), Dense 128, SE ratio 8, augmentation ±10° / ±10% shift / ±10% zoom, seeds [42] |
<!--END:hyperparameters-->

## 5. Hyperparameter Tuning
- **Parameters considered / values tested:** learning rate {3e-4, 1e-3} × dropout {0.3, 0.5}, batch size fixed at 64 (4 configurations, proposed model, one seed, 4 epochs each because of limited CPU compute).
- **Selection rule:** highest **validation** accuracy (ties → lower validation loss). The test set was never used for tuning.
- **Final values:** row marked `yes` below; the same values are used for every model in the ablation and comparison so that the comparison is fair.
- **Reason:** [EDIT AFTER RUNNING — one or two sentences, e.g. why the selected learning rate/dropout/batch size trained most stably].

<!--BEGIN:tuning-->
| lr | dropout | batch_size | val_accuracy | val_loss | epochs_run | selected |
|---|---|---|---|---|---|---|
| 0.001 | 0.3 | 64 | 0.98567 | 0.11984 | 4 | yes |
| 0.0003 | 0.3 | 64 | 0.98233 | 0.14321 | 4 |  |
| 0.0003 | 0.5 | 64 | 0.98067 | 0.16864 | 4 |  |
| 0.001 | 0.5 | 64 | 0.96617 | 0.19449 | 4 |  |
<!--END:tuning-->

## 6. Results
Test set = 10,000 official MNIST test images. Figures: `figures/confusion_matrix.png`, `figures/confusion_matrix_baseline.png`, `figures/training_curve.png`, `figures/training_curve_baseline.png`, `figures/convergence_comparison.png`. CSVs: `results/results.csv`, `results/per_class_metrics.csv`, `results/confusion_matrix.csv`.

<!--BEGIN:results_main-->
| Model | Accuracy | Precision (macro) | Precision (weighted) | Recall (macro) | Recall (weighted) | F1 (macro) | F1 (weighted) | Specificity (macro) | ROC-AUC (macro OvR) | Test loss | Misclassified |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline CNN | 99.34 | 99.34 | 99.34 | 99.33 | 99.34 | 99.34 | 99.34 | 99.93 | 0.99997 | 0.0305 | 66 |
| Proposed (BN + SE + Aug) | 99.23 | 99.23 | 99.23 | 99.23 | 99.23 | 99.23 | 99.23 | 99.91 | 0.99997 | 0.0437 | 77 |
<!--END:results_main-->

**Per-class metrics (proposed model)**

<!--BEGIN:results_per_class-->
| Class | Precision | Recall | F1 | Specificity | ROC-AUC (OvR) | Support |
|---|---|---|---|---|---|---|
| 0.0 | 0.9939 | 0.9949 | 0.9944 | 0.9993 | 1.0 | 980.0 |
| 1.0 | 0.9921 | 0.9912 | 0.9916 | 0.999 | 1.0 | 1135.0 |
| 2.0 | 0.9894 | 0.9952 | 0.9923 | 0.9988 | 1.0 | 1032.0 |
| 3.0 | 0.9882 | 0.997 | 0.9926 | 0.9987 | 1.0 | 1010.0 |
| 4.0 | 0.9939 | 0.998 | 0.9959 | 0.9993 | 1.0 | 982.0 |
| 5.0 | 0.9921 | 0.991 | 0.9916 | 0.9992 | 0.9999 | 892.0 |
| 6.0 | 0.9968 | 0.9875 | 0.9921 | 0.9997 | 1.0 | 958.0 |
| 7.0 | 0.9932 | 0.9893 | 0.9912 | 0.9992 | 1.0 | 1028.0 |
| 8.0 | 0.9858 | 0.9969 | 0.9913 | 0.9984 | 0.9999 | 974.0 |
| 9.0 | 0.998 | 0.9822 | 0.99 | 0.9998 | 1.0 | 1009.0 |
<!--END:results_per_class-->

**Training analysis (seed 42)**

<!--BEGIN:training_analysis-->
| Model | Epochs run | Best epoch (min val loss) | Train acc @best (%) | Val acc @best (%) | Train-val acc gap (pp) | Min val loss |
|---|---|---|---|---|---|---|
| Baseline CNN | 30 | 29 | 99.9 | 99.13 | 0.76 | 0.0401 |
| Proposed (BN + SE + Aug) | 30 | 26 | 98.94 | 99.05 | -0.11 | 0.0487 |
<!--END:training_analysis-->

**Computational cost**

<!--BEGIN:cost-->
| Model | Trainable params | Total params | Epochs run | Train time (s) | Sec/epoch | Inference (ms/image) |
|---|---|---|---|---|---|---|
| Baseline CNN | 225034 | 225034 | 30 | 500.4 | 16.68 | 0.0857 |
| Proposed (BN + SE + Aug) | 226518 | 226710 | 30 | 940.2 | 31.34 | 0.1406 |
| LeNet-5 | 61706 | 61706 | 16 | 143.0 | 8.93 | 0.0363 |
| VGG-style CNN | 467818 | 467818 | 10 | 608.1 | 60.81 | 0.2363 |
<!--END:cost-->

[EDIT AFTER RUNNING — 3–4 sentences: convergence behaviour, train/val gap, whether the model over/under-fits, which digit pairs are confused most (see confusion matrix).]

## 7. Comparative Analysis (CO4)
**(a) Models re-trained by me on the same dataset, same split, same hyperparameters, same test set** (`results/comparison.csv`; Precision/Recall/F1 are macro-averaged):

<!--BEGIN:comparison-->
| Model | Accuracy | Precision | Recall | F1 | AUC | Parameters |
|---|---|---|---|---|---|---|
| LeNet-5 | 98.92 | 98.92 | 98.91 | 98.91 | 0.99991 | 61706 |
| VGG-style CNN | 99.27 | 99.27 | 99.26 | 99.26 | 0.99996 | 467818 |
| Baseline CNN | 99.34 | 99.34 | 99.33 | 99.34 | 0.99997 | 225034 |
| Proposed (BN + SE + Aug) | 99.23 | 99.23 | 99.23 | 99.23 | 0.99997 | 226710 |
<!--END:comparison-->

**(b) Published results on MNIST** (accuracy only, as reported in the papers; `results/literature_comparison.csv`):

<!--BEGIN:literature-->
| Method | Reference | Reported accuracy (%) | Notes |
|---|---|---|---|
| LeNet-5 | LeCun et al., Proc. IEEE, 1998 | 99.05 | single CNN, no augmentation |
| Multi-column DNN | Ciresan et al., CVPR, 2012 | 99.77 | committee of CNNs, augmentation |
| DropConnect | Wan et al., ICML, 2013 | 99.79 | ensemble of 5 networks |
| CapsNet | Sabour et al., NeurIPS, 2017 | 99.75 | capsule network, shift augmentation |
| Proposed (this work) | - | 99.23 | single model, no ensemble |
<!--END:literature-->

References
1. Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, "Gradient-based learning applied to document recognition", *Proc. IEEE*, 1998.
2. D. Cireşan, U. Meier, J. Schmidhuber, "Multi-column deep neural networks for image classification", *CVPR*, 2012.
3. L. Wan et al., "Regularization of neural networks using DropConnect", *ICML*, 2013.
4. S. Sabour, N. Frosst, G. Hinton, "Dynamic routing between capsules", *NeurIPS*, 2017.
5. K. Simonyan, A. Zisserman, "Very deep convolutional networks for large-scale image recognition", *ICLR*, 2015.
6. J. Hu, L. Shen, G. Sun, "Squeeze-and-Excitation networks", *CVPR*, 2018.
7. S. Ioffe, C. Szegedy, "Batch normalization", *ICML*, 2015.

[EDIT AFTER RUNNING — explain why the proposed model performs better / similarly / worse than each method: e.g. LeNet-5 is smaller and uses tanh + average pooling; the published top results use ensembles or much larger models, whereas the proposed model is a single ~0.23 M-parameter network.]

## 8. Innovation (CO5)
**What is the innovation compared with the baseline / state-of-the-art?**
The baseline CNN is extended with a combination of (A) batch normalisation, (B) squeeze-and-excitation channel attention and (C) geometric data augmentation, while keeping the network almost the same size (+0.7% parameters). BN stabilises and speeds up optimisation, SE lets the network re-weight feature channels, and augmentation targets the main real-world variation in handwriting (slant, position, scale). The contribution is evaluated with (i) an ablation study (single run, seed 42) including leave-one-out variants and (ii) a robustness study under noise, rotation and shift. Unlike most published MNIST gains, the improvement is obtained by a single small model rather than an ensemble.

**Ablation study** (single run, seed 42, so the ± values are 0.00; `results/ablation.csv`, raw runs in `results/ablation_runs.csv`):

<!--BEGIN:ablation-->
| Experiment | BatchNorm | SE attention | Augmentation | F1-macro (%) | Accuracy (%) | Params | Seeds |
|---|---|---|---|---|---|---|---|
| baseline | ✗ | ✗ | ✗ | 99.34 ± 0.00 | 99.34 ± 0.00 | 225034 | 1 |
| +BN | ✓ | ✗ | ✗ | 99.34 ± 0.00 | 99.34 ± 0.00 | 225322 | 1 |
| +BN+SE | ✓ | ✓ | ✗ | 99.41 ± 0.00 | 99.41 ± 0.00 | 226710 | 1 |
| proposed | ✓ | ✓ | ✓ | 99.23 ± 0.00 | 99.23 ± 0.00 | 226710 | 1 |
| proposed - BN | ✗ | ✓ | ✓ | 99.10 ± 0.00 | 99.11 ± 0.00 | 226422 | 1 |
| proposed - SE | ✓ | ✗ | ✓ | 99.25 ± 0.00 | 99.25 ± 0.00 | 225322 | 1 |
| proposed - Aug | ✓ | ✓ | ✗ | 99.41 ± 0.00 | 99.41 ± 0.00 | 226710 | 1 |
<!--END:ablation-->

**Robustness** (seed-42 models; `results/robustness.csv`, `figures/robustness.png`):

<!--BEGIN:robustness-->
| Perturbation | Level | Baseline CNN acc (%) | Proposed (BN + SE + Aug) acc (%) |
|---|---|---|---|
| Clean | 0.0 | 99.34 | 99.23 |
| Gaussian noise (sigma) | 0.1 | 99.31 | 95.29 |
| Gaussian noise (sigma) | 0.2 | 98.83 | 59.09 |
| Gaussian noise (sigma) | 0.3 | 95.24 | 21.27 |
| Rotation (deg) | 10.0 | 98.88 | 99.19 |
| Rotation (deg) | 20.0 | 96.5 | 98.0 |
| Rotation (deg) | 30.0 | 88.59 | 92.16 |
| Horizontal shift (px) | 2.0 | 97.82 | 99.17 |
| Horizontal shift (px) | 4.0 | 79.21 | 97.77 |
<!--END:robustness-->

[EDIT AFTER RUNNING — state which component contributes most, whether the differences exceed the seed-to-seed std, and how the proposed model behaves under perturbation.]

## 9. Folder Structure
```text
RegistrationNumber/
├── README.md
├── requirements.txt
├── data/
│   └── dataset_information.txt       # dataset description (data is downloaded at run time)
├── src/
│   ├── config.py                     # paths, default hyperparameters
│   ├── preprocessing.py              # loading, stratified split, normalisation, dataset tables
│   ├── model.py                      # baseline / proposed / LeNet-5 / VGG-style models
│   ├── train.py                      # training routine, ablation configurations
│   ├── test.py                       # metrics (acc, P, R, F1, specificity, ROC-AUC), confusion matrix
│   ├── tune.py                       # hyperparameter search (validation-based)
│   ├── run_all.py                    # runs everything, writes results/ and figures/, fills README tables
│   └── report.py                     # README table filler + requirements.txt writer
├── notebooks/
│   └── experiments.ipynb    # original baseline experiments (99.07% test accuracy)
├── results/                          # CSV tables (results, per-class, ablation, comparison, tuning, robustness, ...)
├── models/
│   ├── model_description.txt
│   └── *.keras                       # trained models
└── figures/                          # confusion matrices, training curves, class distribution, robustness
```

## Reproduce
    python -m venv .venv
    .venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Linux/Mac)
    pip install tensorflow numpy pandas scikit-learn scipy matplotlib seaborn
    python src/run_all.py --quick --seeds 1     (fast sanity check)
    python src/run_all.py                       (full run; fills results/, figures/, models/, README tables, requirements.txt)
