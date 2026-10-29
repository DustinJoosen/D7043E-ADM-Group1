# 1. Environment
Python 3.9+ is recommended. From the project folder, install the dependencies from requirements.txt:

`python -m pip install --upgrade pip`

# 2. Getting the raw data

Both scripts read the records through wfdb, in 2 specific ways:
- Stream from PhysioNet (default): needs internet access on every run. Nothing to set up.
- Local copy (recommended for repeat runs): download once, then pass --local_dir.

An online copy of the data can be found [here](https://physionet.org/content/mitdb/1.0.0/).

# 3. Running prepare_data.py
Run this from the project root. It builds the 3 different datasets (train/test/test) from the raw records.

`python prepare_data.py --outdir ./data --pre 90 --post 110 --val_frac 0.2 --seed 42 --local_dir ./mit-bih-arrhythmia-database-1.0.0`

Omit --local-dir to stream the records from PhysioNet instead (needs internet).

| Argument | Default  | Meaning                                         |
|---|----------|-------------------------------------------------|
|`--oudir` | `./data` | Where the outputs are written                   |
|`--local_dir` | none     | Folder with the downloaded ECG records          |
| `--pre` / `--post` | 90 / 110 | Samples kept before / after each R-peak         |
| `--val_frac` | 0.2      | fraction of DS1 RECORDS held out for validation |
| `--seed` | 42       | Seed for the split                              |


Outputs in `./data`:

- `train.npz`, `val.npz`, `test.npz`
- `class_distribution.csv`, `record_distribution.csv`
- `exclusions.json`
- `config.json` 

### Optional: data quality check and EDA notebook

``python3 data_quality_check.py --outdir ./data/quality --local_dir ./mit-bih-arrhythmia-database-1.0.0``
