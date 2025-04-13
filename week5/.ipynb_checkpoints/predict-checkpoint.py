
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


class Customer(BaseModel):
    customerid: str
    gender: str
    seniorcitizen: int
    partner: str
    dependents: str
    tenure: int         
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    monthlycharges: float
    totalcharges: float

def predict(customer: Customer)-> float:
    C=1.0
    numerical = ['tenure', 'monthlycharges', 'totalcharges']
    categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
        'phoneservice', 'multiplelines', 'internetservice',
        'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
        'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
        'paymentmethod']
    fields = categorical + numerical

    filename = f"model_C={C}.bin"

    with open(filename, 'rb') as fl:
        dv, model = pickle.load(fl)

    df_customer = pd.DataFrame([customer])[fields]
    customer_dict = df_customer.to_dict(orient='records')[0]
    X_customer = dv.transform(customer_dict)
    return model.predict_proba(X_customer)[0,1]


app = FastAPI()

@app.post("/predict/")
def is_churn(customer: Customer):
    churn_prob = predict(customer.model_dump())
    prediction = churn_prob >= 0.5
    if prediction:
        prediction = "Yes"
    else:
        prediction = "No"
    return {"prediction": prediction, "probability": churn_prob} 
  

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == '__main__':
    cs = {
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
    "totalcharges": 6770.5
}
    print(predict(cs))