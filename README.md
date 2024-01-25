# Heart Disease Classifier
> This project is in accordance with the requirements for MLZoomcamp course's capstone project.

## Project Description
The goal of this project is to build a webservice that would take the features of persons and classify if that patient has heart disease or not.
The dataset for this project is taken from UCL repository and can be found in the repo.

## The Process
The _**script.py**_ file creates two models model_lr.bin and model_rf.bin for logistic regression and random forest models respectively.
Run predict.py file to run the flask server.
Go to *localhost:9696* and fill the form to check.

## Run with Flask
The flask app can be run independently without docker with:
```
python predict.py
```
test using sending requests to `http://localhost:9696/predict`.

## Run with Docker
To run the app with docker; follow the steps mentioned:  
1. Build the image  
    ```
    docker build -t heart-disease-classifier .
    ```  
2. Run
    ```
    docker run -it -p 9696:9696 heart-disease-classifier:latest
    ```
    This command will run the service at `http://localhost:9696/predict`.

### It can also be run as a html frontend on `http://localhost:9696`
---