# Class Imbalance in Casting Defect Classification (XBDS3024N)

Python implementation for the XBDS3024N Image Processing and Computer Vision
assignment. The project studies how class imbalance affects the classification
of submersible-pump impeller castings as **OK** or **Defective**. It compares
four classical models (SVM, Logistic Regression, k-NN, Random Forest) with two
neural architectures (MobileNetV2 transfer learning, a CNN trained from scratch)
and tests six methods for handling imbalance against a no-handling baseline.


**Live dashboard:** [Streamlit app URL]  
**Source code:** [GitHub repository URL]  
**Dataset:** [Casting Product Image Data for Quality Inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) by Dabhi (2020), CC BY-NC-ND 4.0. The images are not included in this repository. Download them from Kaggle and place them in `data/casting_512x512/`.

## 1. Folder structure

```
casting_project/
├── casting_imbalance/          # reusable, object-oriented package
│   ├── config.py               # every setting in one frozen dataclass
│   ├── data.py                 # CastingDataset, DataSplitter, ImbalanceScenarioBuilder
│   ├── preprocessing.py        # ImagePreprocessor (resize / Gaussian / CLAHE)
│   ├── features.py             # HandcraftedFeatureExtractor (HOG, LBP, GLCM), FeatureBank
│   ├── imbalance.py            # Strategy classes + ThresholdOptimizer
│   ├── evaluation.py           # Evaluator (imbalance-aware metrics)
│   ├── experiment.py           # ExperimentRunner (ratio x repetition x method)
│   ├── inference.py            # deployable pipelines + InspectionService
│   ├── plotting.py             # shared figure style
│   └── models/
│       ├── base.py             # BaseClassifier (abstract interface)
│       ├── classical.py        # FeatureTransformer, SVM, LogisticRegression, KNN, RandomForest
│       └── neural.py           # ImageAugmenter, MobileNetV2Embedder, TransferHead, SimpleCNN
├── scripts/                    # run these in order (see section 3)
├── app.py                      # Streamlit guided learning dashboard (light theme)
├── .streamlit/config.toml      # dashboard theme
├── predict.py                  # command-line inspection
├── data/casting_512x512/       # put the dataset here (def_front/, ok_front/)
├── weights/                    # optional local MobileNetV2 ImageNet weights
├── outputs/                    # tables, figures, logs, cached embeddings
└── models/                     # trained pipelines used by the application
```

## 2. Setting up in Visual Studio Code

1. Install Python 3.10–3.12 and the VS Code *Python* extension.
2. Open the `casting_project` folder in VS Code (*File → Open Folder*).
3. Open a terminal (*Terminal → New Terminal*) and create a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:      .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. Select the interpreter: *Ctrl+Shift+P → Python: Select Interpreter → .venv*.
5. Download the dataset from Kaggle
   (`ravirajsinh45/real-life-industrial-dataset-of-casting-product`) and copy the
   **unaugmented** `casting_512x512` folder so that you have
   `data/casting_512x512/def_front/` (781 images) and
   `data/casting_512x512/ok_front/` (519 images).
6. MobileNetV2 weights: if `weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5`
   is missing, Keras downloads the ImageNet weights automatically the first time.

## Jupyter walkthrough

`Casting_Imbalance_Notebook.ipynb` (project root) walks through EDA, the 70/15/15 split, the imbalance scenarios, preprocessing, feature extraction, the classical and neural models, the handling methods and the final results, in the style of the module tutorials. Open it in VS Code (Jupyter extension) and choose the same Python environment. A full run takes about 5 to 15 minutes on a CPU; set `RUN_CNN = False` to skip the CNN. Regenerate it with `python scripts/build_notebook.py`.

## 3. Reproducing the study

Run from the project root, in this order:

| Step | Command | Output | CPU time (approx.) |
|---|---|---|---|
| 1 | `python scripts/run_eda.py` | dataset figures, split, integrity checks | 1 min |
| 2 | `python scripts/run_classical.py` | SVM + Random Forest runs, ablations | 5 min |
| 2b | `python scripts/run_more_classical.py` | Logistic Regression + k-NN runs, k ablation | 1 min |
| 3 | `python scripts/run_svm_c_ablation.py` | SVM penalty (C) ablation | 1 min |
| 4 | `python scripts/run_mobilenet.py` | MobileNetV2 runs, ablation | 25 min |
| 5 | `python scripts/run_cnn.py` | from-scratch CNN runs | 1.5–2 h (CPU) |
| 6 | `python scripts/analyse_results.py` | statistics, summary tables, result figures | 1 min |
| 7 | `python scripts/train_final_models.py` | deployable models in `models/` | 3 min |

The split (`outputs/tables/data_split.csv`) is created once with seed 42 and
then reused, so every experiment is evaluated on the same 195 test images.

## 4. Running the application (interactive learning dashboard)

```bash
streamlit run app.py
```
A browser tab opens (light theme). The dashboard is a short companion to the report:

| Page | What it shows |
|---|---|
| 🏠 Overview | Dataset, example images, training set at each imbalance ratio, headline result |
| 🔍 Inspector | Classify a test image or an upload with MobileNetV2 and the SVM |
| 📉 Imbalance | How each model degrades from natural data to 20:1 |
| 🛠️ Fixes | Every handling method for every model at a chosen ratio; threshold playground |
| ⚖️ Compare | Head-to-head of two set-ups, heat map of all results, top-10 ranking |

Edit `STUDENT_NAME` and `STUDENT_ID` at the top of `app.py` to show your details in the sidebar.
The app needs the trained models (`models/`) and the result files in `outputs/` (both included);
the test-image picker needs the dataset in `data/`.

Command-line alternative:
```bash
python predict.py data/casting_512x512/def_front/cast_def_0_0.jpeg
python predict.py some_folder --csv results.csv
```
