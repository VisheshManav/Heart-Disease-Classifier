# Heart Disease Classifier
> This project is in accordance with the requirements for MLZoomcamp course's capstone project.

## Project Description
The Heart Disease Prediction App is designed to assist healthcare professionals and individuals in predicting the likelihood of heart disease based on various input parameters. With the prevalence of heart disease being a significant concern globally, this application aims to provide an accessible and reliable tool for early detection and prevention.

Healthcare professionals can integrate the Heart Disease Prediction App into their practice to enhance cardiovascular risk assessment during patient consultations. The app can facilitate proactive interventions and preventive measures tailored to each patient's unique risk profile, ultimately leading to improved patient outcomes and reduced disease burden.

Furthermore, individuals concerned about their heart health can utilize the app as a self-assessment tool, empowering them to take proactive steps towards prevention. By promoting early detection and lifestyle modifications, the app aims to mitigate the incidence and severity of heart disease, ultimately contributing to better population health and well-being.

The path of this project is to build a webservice that would take the features of persons and classify if that patient has heart disease or not.
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