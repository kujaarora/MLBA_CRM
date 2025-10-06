# Machine Learning for Business Applications (MLBA) Client Retention Model (CRM) 

We introduce a novel hybrid model that combines a RetrievalAugmented Generation (RAG) pipeline with a BERT model to extract a ”Relationship Health Score” from client meeting transcripts. This score, integrated with traditional structured data, is used to train an XGBoost classifier. Key findings demonstrate that our proposed solution achieved an F1-Score of 0.75 and an AUC-ROC of 0.88, significantly outperforming a baseline model that relies only on structured data. Most notably, the model increased prediction recall by 27 percent , enhancing the ability to proactively identify at-risk clients and enabling timely intervention to reduce churn.

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


## Dataset Creation:
* Follow the instructions from the mentioned data source [Dataset](https://github.com/microsoft/topic_conversation/) and copy all the json files generated from the source to the [raw data](./data/source) folder of this project.
* Run the [preprocess_tcr_data.py](./preprocess_tcr_data.py) file to create the final transcripts from the raw data for each meeting in a separate file. 


