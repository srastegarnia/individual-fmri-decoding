# Brain Decoding of HCP Tasks in a Dense Individual fMRI Dataset
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.neuroimage.2023.120395-blue)](https://doi.org/10.1016/j.neuroimage.2023.120395)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code accompanying the research paper:

***Rastegarnia, S. et al. (2023). Brain decoding of the Human Connectome Project tasks in a dense individual fMRI dataset. NeuroImage, 283, 120395.***

## 🧠 Overview
This project demonstrates that **highly accurate brain decoding models** can be trained entirely at the individual level using densely sampled fMRI data. 
We benchmarked 9+ machine learning classifiers to decode 21 experimental conditions from the Human Connectome Project (HCP) task battery using one of 
the largest collections of individual multi-task fMRI data available to date (~7 hours per subject).

**Key Technical Achievement:** Our implementation successfully handles the complex challenge of 
**decoding multiple task conditions from single TR (Repetition Time) resolution fMRI data**, achieving remarkable decoding accuracy despite the 
high temporal resolution constraints.

## Key Findings:
- High decoding accuracy (57-67%) using individual models, approaching state-of-the-art group-level models trained on >500x more data
- Multi-Layer Perceptron (MLP) and Graph Convolutional Networks (GCN) were the top-performing models
- Models learned highly subject-specific features that did not generalize well across participants
- Higher-resolution brain parcellations (e.g., Schaefer-1000) generally yielded better results
- Feture importance maps revealed expected brain regions for relevant cognitive domains
- Successful multi-task decoding at single-TR resolution demonstrating fine-grained temporal discrimination capabilities

## 📦 Repository Structure
```text
individual-fmri-decoding/
├── benchmark_models/               # Core scripts for data processing and classical ML decoding
│   ├── benchmark_decoders.py       # Main script to run all benchmark classifiers
│   ├── hcptrt_data_loader.py       # Loads and preprocesses raw fMRI and event data
│   ├── hcptrt_data_prep.py         # Prepares processed data for decoding
│   ├── beta_maps_benchmark.py      # Script for GLM-based beta map generation
│   ├── sanity_check_beta_map.py    # Utility for checking GLM results
│   └── visualization.py            # Functions for plotting results
├── gcn/                            # Graph Convolutional Network implementation
│   ├── test_notebooks/             # Jupyter notebooks for GCN development
│   ├── conectomes_generator.py     # Generates functional connectivity matrices
│   ├── data_concat_windows_gcn.py  # Prepares data windows for GCN input
│   ├── gcn_model.py                # Defines GCN model architectures (PyTorch)
│   ├── graph_construction.py       # Utilities for building brain graphs
│   └── time_windows_dataset.py     # Dataset class for time-windowed data
├── timeseries_benchmark/           # Time series analysis and benchmarking
│   ├── All_tasks_hcptrt_extended_benchmark_model.ipynb
│   ├── All_tasks_hcptrt_restriced_benchmark_model.ipynb
│   ├── benchmark_utils.py          # Utility functions for benchmark analysis
│   ├── tbenchmark_all.ipynb        # Comprehensive time series benchmarking
│   └── outputs/                    # Directory for output files
├── utils.py                        # Shared utility functions
├── requirements.txt                # Python dependencies
└── README.md
```

## 🚀 Getting Started
1. **Prerequisites**
- Python 3.7+
- The dataset: CNeuroMod hcptrt

The dense individual fMRI dataset used in this work is publicly available as part of the CNeuroMod databank. 
Access requires an inter-institutional data transfer agreement. See the [CNeuroMod access instructions](https://www.cneuromod.ca/access/).

2. **Installation**

Clone this repository and install the required Python packages:
```bash
git clone https://github.com/srastegarnia/individual-fmri-decoding.git
cd individual-fmri-decoding
pip install -r requirements.txt
```
## 🔧 Usage
### Pipeline 1: Classical Machine Learning Benchmark
**Data Preparation:**
```bash
cd benchmark_models
# Run data loading and preparation
```

**Running the Benchmark:**
```python
for task_label in task_labels:    
    tpl_mask = path + 'mask_file.nii.gz'
    event_file = path + 'events_file.tsv'
    
    # Read conditions and generate beta maps
    df = utils.new_conditions(path, event_file, task_label)
    conditions = list(set(df.trial_type))
    utils.postproc_task(subject, task_label, conditions, tpl_mask)
    
    # Run decoder
    utils.check_decoding(subject, task_dir, task_label, tpl_mask)
```

## 📊 Outputs
The pipelines generate:
- **Processed Data:** NumPy arrays (.npy) and CSV files
- **Functional Connectomes:** Adjacency matrices
- **Beta Maps:** GLM-generated condition maps
- **Results Summary:** Accuracy metrics
- **Visualizations:** Confusion matrices, training plots
- **Trained Models:** Saved model weights

## 🤝 Contributing
Contributions to improve the code or extend the benchmarks are welcome. Please fork the repository and submit a Pull Request.


## 📜 License
This code repository is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

The associated research paper is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

## 🙏 Acknowledgments
This work is part of the Courtois Project on Neuronal Modeling (CNeuroMod), made possible by a generous donation from the Courtois Foundation.

**Funding:** Courtois Foundation, Fonds de recherche du Québec - Santé (FRQS), Wu Tsai Neurosciences Institute
**Computing:** Digital Research Alliance of Canada (formerly Compute Canada), Calcul Québec
**Data:** The CNeuroMod team and all research participants

## 📫 Contact
- Shima Rastegarnia: srastegarnia@gmail.com

Please cite our work if you use this code:
``` bibtex
@article{Rastegarnia2023,
  title = {Brain decoding of the Human Connectome Project tasks in a dense individual fMRI dataset},
  journal = {NeuroImage},
  volume = {283},
  pages = {120395},
  year = {2023},
  author = {Shima Rastegarnia and Marie St-Laurent and Elizabeth DuPre and Basile Pinsard and Pierre Bellec},
  doi = {https://doi.org/10.1016/j.neuroimage.2023.120395}
}
```

