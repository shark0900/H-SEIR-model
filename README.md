# H-SEIR Model Implementation

This repository contains the implementation code and simulation scripts for the H-SEIR rumor propagation model.

## Description

The H-SEIR (Heterogeneous SEIR) model incorporates individual psychological heterogeneity into rumor propagation dynamics. This repository provides:

1. **Model implementation** - Core H-SEIR ODE system
2. **Parameter fitting scripts** - Fit to real data (PHEME, Weibo)
3. **Simulation scripts** - Numerical experiments
4. **Visualization tools** - Generate publication-quality figures

## Datasets

### 1. PHEME Dataset
- **Description**: Twitter conversations around 9 breaking news events
- **Access**: https://figshare.com/projects/PHEME/17268
- **Used in**: Model validation, parameter estimation

### 2. Weibo Dataset (Chinese_Rumor_Dataset)
- **Description**: Chinese rumor data from Sina Weibo (31,669 rumors + CED extension with 1,538 rumors + 1,849 non-rumors with repost/comment timestamps)
- **Access**: https://gitee.com/sliver-king/Chinese_Rumor_Dataset
- **Used in**: Cross-platform validation, time-series fitting
- **Citation**: If using this dataset, please cite:
  - Liu, Z., Zhang, L., Tu, C. & Sun, M. Statistical and semantic analysis of rumors in Chinese social media. Sci. Sinica Inform. 45, 1536 (2015).
  - Song, C., Tu, C., Yang, C., Liu, Z. & Sun, M. CED: Credible early detection of social media rumors. arXiv:1811.04175 (2018).

## Installation

```bash
# Clone this repository
git clone https://github.com/jinshanzhang/H-SEIR-model.git
cd H-SEIR-model

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Parameter Fitting to Real Data

```bash
# Fit H-SEIR to PHEME dataset
python fit_pheme.py --data ../PHEME/data --output results/pheme_fit.png

# Fit H-SEIR to Weibo dataset (CED_Dataset)
python fit_weibo_real_data.py --data ../Chinese_Rumor_Dataset/CED_Dataset --output results/weibo_fit.png
```

### 2. Run Simulations

```bash
# Basic simulation
python simulate_h_seir.py --N 5000 --beta0 0.5 --gamma 0.1 --theta 0.4 --tau 0.5 --epsilon 0.6

# Time-varying psychological parameters
python simulate_time_varying.py --output results/fig_time_varying.png

# Two-rumor competition
python simulate_two_rumor.py --output results/fig_two_rumor.png
```

### 3. Generate Figures

```bash
# Generate all figures for the paper
python generate_all_figures.py --output ../figures/
```

## File Structure

```
H-SEIR-model/
├── fit_pheme.py                      # Fit to PHEME dataset
├── fit_weibo_real_data.py           # Fit to Weibo dataset (REAL DATA)
├── simulate_h_seir.py               # Basic H-SEIR simulation
├── simulate_time_varying.py         # Time-varying psychological parameters
├── simulate_two_rumor.py           # Two-rumor competition
├── generate_all_figures.py        # Generate publication figures
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── data/                          # Data processing scripts
│   ├── extract_weibo_time_series.py   # Extract time-series from Weibo data
│   └── preprocess_pheme.py            # Preprocess PHEME data
├── results/                       # Fitting results
│   ├── weibo_fit_results.json        # Real fitting results (R² = 0.7708)
│   └── pheme_fit_results.json        # PHEME fitting results (R² = 0.924)
└── figures/                      # Generated figures
    ├── Fig_Weibo_Fit.png           # Weibo fitting figure
    ├── Fig_TimeVarying_Params.png  # Time-varying parameters
    └── Fig_TwoRumor_Competition.png # Two-rumor competition
```

## Real Data Fitting Results

### Weibo Dataset (CED_Dataset)
- **Average R²**: 0.7708 (range: 0.1643 - 0.9856)
- **Number of rumors fitted**: 10 (out of 1,538 total)
- **Fitting method**: Levenberg-Marquardt algorithm
- **Details**: See `fit_weibo_real_data.py` and `results/weibo_fit_results.json`

### PHEME Dataset
- **Average R²**: 0.924
- **Number of events**: 9
- **Details**: See `fit_pheme.py`

## Requirements

- Python 3.8+
- numpy
- scipy
- matplotlib
- python-docx (for Word document processing)

## Citation

If you use this code or model, please cite:

```bibtex
@article{zhang2026hseir,
  title={H-SEIR: Incorporating Individual Psychological Heterogeneity into Rumor Propagation Dynamics},
  author={Zhang, Jinshan},
  journal={Physica A},
  year={2026},
  note={Under review}
}
```

## License

This code is released under the MIT License (see LICENSE file).

## Contact

For questions or bug reports, please open an issue on this repository.

---

**Important Note**: This repository contains only the model implementation and processing scripts. The raw datasets (PHEME, Weibo) must be downloaded from the provided links due to terms of use.
