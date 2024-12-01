# Hickathon_43

Repository of groupe 43 for the Hickathon 2024 #5

Link to the Hackathon repository: https://github.com/hi-paris/Hickathon5

## Team members
- [Ahmed-Yassine CHRAA](ENSTA Paris)
- [Emma De Charry](ENSTA Paris)
- [Florian Morel](ENSTA Paris)
- [Lucien Perderix](ENSTA Paris)
- [Matteo Denis](ENSTA Paris)
- [Yael Gossec](ENSTA Paris)

## Hackathon subject
The subject of the hackathon is to create a model that can predict the level of ground water in a given area. 

## Repository structure
```bash

├───data
│   ├───cleaned
│   │   └───pipelines
│   ├───processed
│   │   └───pipelines
│   ├───src
│   └───submissions
├───notebooks
```

## Approach to the problem
1. Data cleaning & preprocessing
We created a pipeline to clean and preprocess the data. The pipeline is saved in the `data/cleaned/pipelines` folder
All transformations are coded in classes inside the `transformers.py` file and then added to the pipeline in the `pipe.py` file
4. Model selection and fine-tuning
We used a Random Forest Classifier, XGBoost and a CatBoost model to predict the level of ground water. We then used optuna to fine-tune the hyperparameters of each model
We also tried combining the three models using a voting classifier 

## How to run the code
1. Clone the repository
2. Download the data from the hackathon repository and place it in the `data` folder
3. Run processing_clean.ipynb to clean and preprocess the data and chose a model to train
4. Run submission_clean.ipynb to make a submission

You can use optuna files to fine-tune the hyperparameters of a model
