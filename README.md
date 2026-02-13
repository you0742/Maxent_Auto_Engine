MaxEnt Auto Engine v1.4.9-R3 [Build 260213]
MaxEnt Auto Engine is a high-performance Python-based automation framework designed for large-scale Species Distribution Modeling (SDM). It streamlines the entire research pipeline—from baseline model training to future climate projections and ensemble statistical analysis—ensuring rigorous data integrity and computational stability.

✨ Key Features
Automated Research Pipeline: Seamless execution of Phase 1 (Baseline Training), Phase 2 (Future Projection), and Phase 3 (Statistical Averaging).

Data Integrity (SHA-256): Features a unique "Model Fingerprinting" system that uses SHA-256 hashing to ensure projection parameters strictly align with the original training configuration.

Concurrency & Stability: Implements Manager.Lock mechanisms to prevent I/O conflicts and Java Core crashes during multi-processed executions.

Memory-Efficient Processing: Employs Tile-based Streaming for Phase 3, allowing for the processing of high-resolution ASCII grids (1.2GB+) without Out-of-Memory (OOM) errors.

User-Friendly GUI: Provides a localized Korean Graphical User Interface (GUI) for intuitive operation by ecological researchers.

📂 Directory & Data Specification
The engine strictly adheres to the following hierarchical structure for automated data mapping and processing:

0. Species Occurrence Data
Path: {base_dir}/01.SpeciesDATA/*.csv

Format: CSV (Tab delimited).

Column Requirements: - SPECIES: Name of the target species.

LAT: Latitude (Decimal Degrees, WGS84).

LON: Longitude (Decimal Degrees, WGS84).

1. Input (Environmental Layers) — [Format: ESRI ASCII Required]
All environmental variables (e.g., Bioclim, Topography) must be provided in ESRI ASCII (.asc) format.

Phase 1 (Baseline Training): {base_dir}/02.ASCII_DATA/GCM-Current/SSP-Current/Present/*.asc

Note: Predictors used here will define the model's parameters.

Phase 2 (Future Projection): {base_dir}/02.ASCII_DATA/GCM-Future/{GCM}/{SSP}/{Period}/*.asc

Note: File names in each scenario folder must strictly match those used in Phase 1.

⚠️ Importance of Consistency: The engine requires consistent spatial extents, resolutions, and headers across all baseline and projection layers to maintain statistical validity.

2. Output (Results) — [Auto-Created]
The engine ensures that the output directory structure perfectly mirrors the input scenario for systematic data management.

Phase 1 Artifacts: {base_dir}/03.Results/{species}/GCM-Current/SSP-Current/Present

Contains .lambdas and model_meta.json.

Phase 2 Projections: {base_dir}/03.Results/{species}/GCM-Future/{GCM}/{SSP}/{Period}

Contains projected .asc files.

Phase 3 Statistics: - {species}_{period}_[avg/max/min/median/stddev].asc

Auto-generated within the Phase 2 folders.

🚀 Installation & Usage
Prerequisites
Java Runtime Environment (JRE): Necessary for running the MaxEnt core logic.

Python 3.8+

Setup

# Clone the repository
git clone https://github.com/you0742/MaxEnt_Auto_Engine.git


## 🛠 Requirements

### 1. Programming Language
- **Python 3.8 or higher**

### 2. Python Dependencies
- **NumPy**: Used for high-speed statistical calculations in Phase 3.
  ```bash

pip install numpy


Execution
python maxent_engine_refactor_v3.py



📂 Directory Structure
To ensure the engine functions correctly, please organize your workspace according to the hierarchy below. The engine is designed to auto-generate the corresponding subdirectories within 03.Results to match your input parameters.

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