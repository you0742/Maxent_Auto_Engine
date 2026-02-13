# MaxEnt Auto Engine v1.5 [Build 260214]

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Field: Mammalogy](https://img.shields.io/badge/Field-Mammal%20Ecology-green.svg)]()

**MaxEnt Auto Engine** is a high-performance Python-based automation framework designed for large-scale **Species Distribution Modeling (SDM)**. It streamlines the entire research pipeline—from baseline model training to future climate projections and ensemble statistical analysis—ensuring rigorous data integrity and computational stability.

---

## ✨ Key Features

* **Automated Research Pipeline:** Seamless execution of Phase 1 (Baseline Training), Phase 2 (Future Projection), and Phase 3 (Statistical Averaging).
* **Data Integrity (SHA-256):** Features a unique "Model Fingerprinting" system using SHA-256 hashing to ensure projection parameters strictly align with the original training configuration.
* **Concurrency & Stability:** Implements `Manager.Lock` mechanisms to prevent I/O conflicts and Java Core crashes during multi-processed executions.
* **Memory-Efficient Processing:** Employs **Tile-based Streaming** for Phase 3, allowing for the processing of high-resolution ASCII grids (1.2GB+) without Out-of-Memory (OOM) errors.
* **User-Friendly GUI:** Provides a localized Korean Graphical User Interface (GUI) for intuitive operation by researchers.

---

## 📂 Directory Structure

To ensure the engine functions correctly, please organize your workspace according to the hierarchy below. The engine is designed to **auto-generate** the corresponding subdirectories within `03.Results`.



```text
MaxEnt-Auto-Engine/
├── maxent_engine_refactor_v3.py      # Main Python Script
├── maxent.jar                        # MaxEnt Core (Must be present)
├── maxent_Engine_config.ini          # Configuration file (Auto-generated)
├── 01.SpeicesLIST/                   # [Input] Species occurrence data
│   ├── Panthera_tigris.csv           # Tab-delimited (SPECIES, LAT, LON)
│   └── Lynx_lynx.csv
├── 02.ASCII_DATA/                    # [Input] Environmental ASCII layers (.asc)
│   ├── GCM-Current/
│   │   └── SSP-Current/
│   │       └── Present/              # Baseline environmental variables
│   └── GCM-Future/
│       └── {GCM_Name}/               # e.g., UKESM1-0-LL
│           └── {SSP_Scenario}/       # e.g., SSP245, SSP585
│               └── {Period}/         # e.g., 2041-2060
└── 03.Results/                       # [Output] Auto-generated results
    └── {Species_Name}/
        ├── GCM-Current/
        │   └── SSP-Current/
        │       └── Present/          # Phase 1: Model (.lambdas), Metadata
        └── GCM-Future/               # Phase 2 & 3: Projections & Ensemble Stats
            └── {GCM_Name}/
                └── {SSP_Scenario}/
                    └── {Period}/     # Projected probability maps (.asc)


```
## 🚀 Installation & Usage
Prerequisites
The following software environments are required to run the engine:
Java Runtime Environment (JRE): Essential for executing the MaxEnt core logic (maxent.jar).
Python 3.8 or higher: The primary environment for the engine's main scripts.

Setup & Requirements
Clone the repository and install the necessary library for statistical analysis (Phase 3).

## 1. Clone the repository
```text
git clone https://github.com/you0742/MaxEnt_Auto_Engine.git
```

## 2. Navigate to the project directory
Clone the repository and install the necessary library for statistical analysis (Phase 3).
```text
cd MaxEnt_Auto_Engine
```

## 3. Install required library (NumPy)
```text
pip install numpy
```

## 4. Execution
Once the data is properly placed in the designated directories, run the engine using the following command:
```text
python maxent_engine_refactor_v3.py
```

## Change Log
v1.5 Bug Fix

