# MACE-MD-TEST

Running MD simulations via MACE, basic plotting and analysis.

## Directory Structure

```text
mace-md-test/
├── data/
│   └── raw/              # Raw input structures
├── results/              # Output data, MD trajectories and plots
├── tests/                # Scripts for verifying execution and linting
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
└── run_reproduction.sh   # Bash script to execute the full test pipeline
```

## Usage

### Run Reproduction Pipeline
The primary MD reproduction suite, which includes running 2ps MD on two models (medium-mpa-0 and medium-ob3) and plotting them versus temperature, pressure.
Execute the shell script:

```text
bash run_reproduction.sh
```
### Restartable MD simulation
Execute MD in mutiple runs by `tests/run_md_auto.py`.
Once all simulations are finished, merge the data by `md_csv_merge.py`



