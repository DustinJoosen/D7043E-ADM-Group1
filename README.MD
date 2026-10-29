# 1. Environment
Python 3.9+ is recommended. From the project folder, install the dependencies from requirements.txt:

`python -m pip install -r requirements.txt`

# 2. Setup

**Note**: This step can be ignored if the data folder already exists

## 2.1. Prepare data
`python prepare_data.py` should be run if there are no `*.npz` files in the data folder. The script grabs the data online, and builds the 3 different datasets (train, val, test) from those raw records.

By default they get added to the data folder. If a different folder is desired, the `--outdir ./data` option can overwrite this. Keep in mind that most scripts assume the data folder.


| Argument | Default  | Meaning                                         |
|---|----------|-------------------------------------------------|
|`--outdir` | `./data` | Where the outputs are written                   |
| `--pre` / `--post` | 90 / 110 | Samples kept before / after each R-peak         |
| `--val_frac` | 0.2      | fraction of DS1 RECORDS held out for validation |
| `--seed` | 42       | Seed for the split                              |

Outputs in `./data/`:

- `test.npz`
- `training.npz`
- `val.npz` 
- `class_distribution.csv`
- `record_distribution.csv`
- `exclusions.json`
- `config.json` 


## 2.2. Data quality check

`python data_quality_check.py` scans the raw MIT-BIH records (before filtering or beat extraction) and reports missing data, extreme amplitudes, flat segments, noisy windows, and differences between recordings and ECG channels.

By default the reports are written to `/data/quality`. If a different folder is desired, the `--outdir ./data/quality` option can overwrite this. 

| Argument | Default | Meaning |
|---|---|---|
| `--outdir` | `./data/quality` | Where the quality reports are written |
| `--local_dir` | none | Folder with the downloaded ECG records; omit to stream from PhysioNet |
| `--records` | all 48 | Specific record IDs to check, e.g. `--records 100 101` |


Outputs in `./data/quality`:

- `data_quality_report.csv`
- `flagged_issues.json`
- `lead_usage_summary.json`
