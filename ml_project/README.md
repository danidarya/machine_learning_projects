# Homework 01

Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
Usage:
~~~
python src/train_and_predict_pipeline.py train configs/config_logreg.yaml
python src/train_and_predict_pipeline.py train configs/config_random_forest.yaml
~~~

Predict:
~~~
python src/train_and_predict_pipeline.py predict configs/config_logreg.yaml
python src/train_and_predict_pipeline.py predict configs/config_random_forest.yaml
~~~

Test:
~~~
pytest tests
~~~


Project Organization
------------


    ├── configs            <- configs for logistic regression and random forest pipelines
    │         
    │  
    ├── data_raw           <- directory for data for train(heart.csv) and data to be predicted 
    │   
    │
    ├── logs               <- model logs
    │
    ├── models             <- Trained and serialized models, model predictions, metrics
    │
    ├── notebooks          <- Jupyter notebook with data analysis.
    │                                                 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── data           <- code to download or generate data
    │   │
    │   ├── features       <- code to turn raw data into features for modeling
    │   │
    │   ├── models         <- code to train models and then use trained models to make
    |   |
    │   └── params         <- code to define split,feature and model parameters   
    |
    └── tests              <- tests for dataset,features and train pipeline


--------
Выполнены все пункты задания, кроме 8 (кастомный трансформер), 11 (hydra), 12 (github actions) суммарно на 27 баллов + 1 за самооценку. 
