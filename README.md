# South Sudan Project 
## Code description

This repository contains all the code for the experiment that group 25 did. The data folder contains the following:
- The folder `articles_ratings`: this folder contains the results from the manual labelling experiment.
- `articles_topics.csv`: contains all that went through the topic moddeling process. All the categories are also split by sentiment
- `food_crises_cleaned.csv`: contains all the ipc scores that are used during the prediction process.

The home folder contains all the scripts and notebooks that are needed to replicate all the experiments.

## Requirements
To install the requirements open Terminal (macOS)/Command Prompt (Windows) and run pip install -r requirements.txt. If you create a new environment in PyCharm, an icon should appear to install requirements. The code runs with Python 3.9.0.

Required libraries:
- bertopic == 0.15.0 
- pandas == 1.4.4 
- geopandas == 0.13.2 
- matplotlib == 3.7.2 
- seaborn == 0.12.2
- statsmodels == 0.14.0 

## Recomputing sentiment

This can be done by running `create_sentiment.py`. This uses multiprocessing so it creates multiple csv files after the whole process is done. These csvs can be combined by running the `combine.py` script.

## Recreating all the topics

This can be done by running the `topic_moddeling.ipynb` notebook from top to bottom. To create the topic modelling using a different model, simple remove `southsudan_model` from the folder and run all the cells. This will create a new model. This performs the topic modeling process and it for each topic it creates three seperate columns, namely `<topic>_positive`, `<topic>_neutral`, `<topic>_negative`. 

## Recreating the results from the manual labelling experiment

This can be done by running the `experiment_results.ipynb` notebook. This performs a spearman's rank correlation test for all the possible combinations and it does a permutation test, because of the small sample size. All the cells will print out  the correlation between the two tested variables.

## The baseline model

The notebook `baseline_model.ipynb` contains the baseline implementation of the ipc prediction model. This model is used as a comparison to see if our methods improved the performance of the baseline model.

## The final model

The notebook `final_notebook.ipynb` contains the final model we created, using the seperate topics. Simply run all the cells from top to bottom to get the final performance of the model. The final evaluation metrics are printed at the bottom of the final cell.

## Computing the bias for each country

This can be done by running `bias_of_neighbouring_countries.py`. This will print out the proportion of positive articles for each of the neighbouring countries. 