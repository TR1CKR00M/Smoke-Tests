# dualx

Project files are organized by purpose:

```text
dualx/
├── data/
│   ├── raw/          # Original, unchanged input data
│   ├── interim/      # Temporary or partially processed data
│   └── processed/    # Cleaned data ready for analysis
├── docs/             # Project documentation and notes
├── results/
│   ├── figures/      # Generated plots and images
│   ├── reports/      # Analysis reports and summaries
│   └── tables/       # Exported result tables
├── src/              # Reusable project code
├── tests/            # Automated tests
└── main.py           # Project entry point
```

Keep raw inputs unchanged and place generated artifacts in `results/`. Avoid
committing large or sensitive datasets; document how to obtain them when needed.
