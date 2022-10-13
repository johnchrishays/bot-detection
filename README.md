# Simplistic Collection and Labeling Practices Limit the Utility of Benchmark Datasets for Twitter Bot Detection

## Overview

This repository contains code and data produced for the analysis we carried out for ``Simplistic Collection and Labeling Practices Limit the Utility of Benchmark Datasets for Twitter Bot Detection''. It is necessary to download the datasets described in the paper to replicate the analysis.

## Directory overview

| Folder or Filename                         | Description                                                                                                                                                                                                                                               |
| :----------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Simplistic Collection and Labeling Practices Limit the Generalizability of Benchmark Datasets for Twitter Bot Detection.ipynb`                    | Notebook for running experiments and visualizing results.
| `gen_tables.ipynb`                    | Notebook for generating the tables used in the paper.
| `data`                    | Outputs of analysis code.
| `data_accessor.py`                    | Utilities for loading datasets.
| `fit_and_score.py`                    | Utilities for fiting and scoring models.
| `preprocess.py`                    | Utilities for preprocessing data used in `data_accessor`.
| `print_table.py`                    | Utilities for printing latex-ready tables.
| `train_on_one_test_on_another.py`                    | Utilities for training on one dataset and testing on another.
| `leave_one_dataset_out.py`                    | Utilities for experiments training on all but one dataset and leaving one out.




## Setup

Install needed packages:
```
python3 -m pip install -r requirements.txt
```

Some of the data used in this paper is available in the [OSoMe Bot Repository](https://botometer.osome.iu.edu/bot-repository/). For other datasets, it will be necessary to contact the authors of the dataset's originating paper.

## Run Analysis

Run jupyter notebook:
```
jupyter notebook
```
and select `Simplistic Collection and Labeling Practices Limit the Generalizability of Benchmark Datasets for Twitter Bot Detection.ipynb` for analysis or `gen_tables.ipynb` to generate the tables in the analysis.



