# Machine Learning for Business Applications (MLBA)
# Client Retention Model (CRM) 

## Dataset Creation:
* Follow the instructions from the mentioned data source [Dataset](https://github.com/microsoft/topic_conversation/) and copy all the json files generated from the source to the [raw data](./data/source) folder of this project.
* Run the [preprocess_tcr_data.py](./preprocess_tcr_data.py) file to create the final transcripts from the raw data for each meeting in a separate file. 


## Setup:
* Create a venv with (crm) in python:
```
python -m venv crm
```
* Now activate the virtual environment:
```
source crm/bin/activate
```

* Run below command to install all the necessary dependencies in the virtual environment:
```
pip install -r requirements.txt
```


