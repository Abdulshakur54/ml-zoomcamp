import requests
import json

customer = {
    "customerid": "2898-mrkpi",
    "gender": "male",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "yes",
    "tenure": 68,
    "phoneservice": "yes",
    "multiplelines": "yes",
    "internetservice": "fiber_optic",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "yes",
    "streamingmovies": "yes",
    "contract": "one_year",
    "paperlessbilling": "yes",
    "paymentmethod": "credit_card_(automatic)",
    "monthlycharges": 101.05,
    "totalcharges": 6770.5,
}
url = "http://127.0.0.1:8000/"
customer_json = json.dumps(customer)
rsp = requests.post(url, data=json.load(customer_json))
print(rsp.json())
