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
- **Access**: https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078
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
git clone https://github.com/shark0900/H-SEIR-model.git
cd H-SEIR-model

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Parameter Fitting to Real Data

```bash
# Fit H-SEIR to Weibo dataset (CED_Dataset)
python fit_h_seir_real_data.py --output results/weibo_fit.png

# Extract time-series from Weibo data
python extract_weibo_time_series.py --output results/
```

### 2. Run Simulations

```bash
# PHEME calibration simulation
python sim_pheme_calibration.py --output results/pheme_calibration.png

# Time-varying psychological parameters
python sim_time_varying_params.py --output results/fig_time_varying.png

# Two-rumor competition
python sim_two_rumor.py --output results/fig_two_rumor.png
```

## File Structure

```
H-SEIR-model/
├── extract_weibo_time_series.py     # Extract time-series from Weibo CED data
├── fit_h_seir_real_data.py          # Fit H-SEIR to real Weibo data
├── sim_pheme_calibration.py         # PHEME dataset calibration simulation
├── sim_time_varying_params.py       # Time-varying psychological parameters
├── sim_two_rumor.py                 # Two-rumor competition simulation
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── data/                            # Data directory
├── results/                         # Fitting results
│   ├── weibo_fit_results.json       # Real fitting results (R² = 0.7708)
│   └── weibo_stats.json             # Weibo dataset statistics
└── figures/                         # Generated figures
    └── Fig_Weibo_Fit.png            # Weibo fitting figure
```

## Real Data Fitting Results

### Weibo Dataset (CED_Dataset)
- **Average R²**: 0.7708 (range: 0.1643 - 0.9856)
- **Number of rumors fitted**: 99 (from 1,538 total in CED)
- **Total time-series records**: 4,433
- **Fitting method**: Levenberg-Marquardt algorithm
- **Details**: See `fit_h_seir_real_data.py` and `results/weibo_fit_results.json`

### PHEME Dataset
- **Average R²**: 0.924
- **Number of events**: 9
- **Details**: See `sim_pheme_calibration.py`

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
