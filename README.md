# ML_project_classifier_target_action
**Project objective:** To create a model for predicting the target action for the company "SberAvtopodpiska".

**"SberAvtopodpiska"** is a long-term car rental service for individuals, which offers a new way of owning a car for the Russian market and acts as an alternative to a car loan.

This company has its own website, which contains a catalog of the cars offered. On the website, the user can perform target actions ("Leave a request", "Order a call"), or non-target actions, such as viewing car cards or "wandering" through the pages of the site. This site collects a wide range of customer data using Google Analytics.

### Project objectives:
1. Conduct Exploratory Data Analysis (EDA).
2. Predict the completion of a target action (estimated ROC-AUC value ~ 0.65).
3. Packaging the model into a service that will take all attributes as input, such as utm_*, device_*, geo_*, and output 0/1 (1 — if the user performs any target action).

### Initial data:
1. File "GA_Sessions.csv" with a data of the visit date and time, description of the client device, etc.
2. File "GA_Hits.csv" with a description of the completed event (target/non-target).

### Main stages of the project:
1. Perform EDA in Jupyter Notebook. - !!! EDA.ipynb !!!
2. Perform preparation of the initial data (cleaning, formatting, standardization and encoding), selection, training and evaluation of the quality of the model in Jupyter Notebook using classical ML algorithms. - !!! Modeling.ipynb !!!
3. Packaging the model into a service by wrapping the process of creating the model in PipeLine with the ability to access the model as an API (using FastAPI). - !!! pipeline_as_a_service !!!
