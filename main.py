from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models_path = os.path.join(os.path.dirname(__file__), 'models')

with open(f'{models_path}/lc_xgboost.pkl', 'rb') as f:
    lc_xgb = pickle.load(f)
with open(f'{models_path}/lc_logistic_regression.pkl', 'rb') as f:
    lc_lr = pickle.load(f)
with open(f'{models_path}/lc_scaler.pkl', 'rb') as f:
    lc_scaler = pickle.load(f)
with open(f'{models_path}/lc_feature_names.pkl', 'rb') as f:
    lc_features = pickle.load(f)
with open(f'{models_path}/hc_xgboost.pkl', 'rb') as f:
    hc_xgb = pickle.load(f)
with open(f'{models_path}/hc_logistic_regression.pkl', 'rb') as f:
    hc_lr = pickle.load(f)
with open(f'{models_path}/hc_scaler.pkl', 'rb') as f:
    hc_scaler = pickle.load(f)
with open(f'{models_path}/hc_feature_names.pkl', 'rb') as f:
    hc_features = pickle.load(f)

print("✅ All models loaded successfully!")


class LendingClubInput(BaseModel):
    loan_amnt: float
    int_rate: float
    installment: float
    annual_inc: float
    dti: float
    revol_bal: float
    revol_util: float
    total_acc: float
    emp_length: float
    home_ownership: int
    purpose: int
    term: float
    model_type: str = "xgboost"


class HomeCreditInput(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    AGE_DAYS: float
    DAYS_EMPLOYED: float
    DAYS_ID_PUBLISH: float
    CNT_CHILDREN: float
    CNT_FAM_MEMBERS: float
    REGION_POPULATION_RELATIVE: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    CODE_GENDER: int
    FLAG_OWN_CAR: int
    FLAG_OWN_REALTY: int
    NAME_CONTRACT_TYPE: int
    NAME_EDUCATION_TYPE: int
    NAME_FAMILY_STATUS: int
    NAME_INCOME_TYPE: int
    model_type: str = "xgboost"


def to_python_float(val):
    return float(val)

def to_python_int(val):
    return int(val)

def get_risk_label(score):
    if score < 30:
        return "Low Risk"
    elif score < 60:
        return "Medium Risk"
    else:
        return "High Risk"

def get_decision(score):
    if score < 30:
        return "Approve"
    elif score < 60:
        return "Manual Review"
    else:
        return "Reject"


@app.get("/")
def root():
    return {"message": "Loan Prediction API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict/lending-club")
def predict_lending_club(data: LendingClubInput):
    input_df = pd.DataFrame([{
        'loan_amnt': data.loan_amnt,
        'int_rate': data.int_rate,
        'installment': data.installment,
        'annual_inc': data.annual_inc,
        'dti': data.dti,
        'revol_bal': data.revol_bal,
        'revol_util': data.revol_util,
        'total_acc': data.total_acc,
        'emp_length': data.emp_length,
        'home_ownership': data.home_ownership,
        'purpose': data.purpose,
        'term': data.term
    }])

    if data.model_type == "logistic":
        input_scaled = lc_scaler.transform(input_df)
        proba = to_python_float(lc_lr.predict_proba(input_scaled)[0][1])
    else:
        proba = to_python_float(lc_xgb.predict_proba(input_df)[0][1])

    risk_score = to_python_int(round(proba * 100))

    feature_importance = {
        'int_rate': round(float(data.int_rate) * 0.5, 2),
        'dti': round(float(data.dti) * 0.16, 2),
        'term': round(float(data.term) * 0.15, 2),
        'loan_amnt': round(float(data.loan_amnt) * 0.001, 2),
        'annual_inc': round((1 / max(float(data.annual_inc), 1)) * 1000, 2)
    }

    return {
        "dataset": "Lending Club",
        "model_used": data.model_type,
        "risk_score": risk_score,
        "default_probability": round(proba * 100, 2),
        "risk_label": get_risk_label(risk_score),
        "decision": get_decision(risk_score),
        "top_factors": feature_importance
    }


@app.post("/predict/home-credit")
def predict_home_credit(data: HomeCreditInput):
    input_df = pd.DataFrame([{
        'AMT_INCOME_TOTAL': data.AMT_INCOME_TOTAL,
        'AMT_CREDIT': data.AMT_CREDIT,
        'AMT_ANNUITY': data.AMT_ANNUITY,
        'AMT_GOODS_PRICE': data.AMT_GOODS_PRICE,
        'AGE_DAYS': data.AGE_DAYS,
        'DAYS_EMPLOYED': data.DAYS_EMPLOYED,
        'DAYS_ID_PUBLISH': data.DAYS_ID_PUBLISH,
        'CNT_CHILDREN': data.CNT_CHILDREN,
        'CNT_FAM_MEMBERS': data.CNT_FAM_MEMBERS,
        'REGION_POPULATION_RELATIVE': data.REGION_POPULATION_RELATIVE,
        'EXT_SOURCE_1': data.EXT_SOURCE_1,
        'EXT_SOURCE_2': data.EXT_SOURCE_2,
        'EXT_SOURCE_3': data.EXT_SOURCE_3,
        'CODE_GENDER': data.CODE_GENDER,
        'FLAG_OWN_CAR': data.FLAG_OWN_CAR,
        'FLAG_OWN_REALTY': data.FLAG_OWN_REALTY,
        'NAME_CONTRACT_TYPE': data.NAME_CONTRACT_TYPE,
        'NAME_EDUCATION_TYPE': data.NAME_EDUCATION_TYPE,
        'NAME_FAMILY_STATUS': data.NAME_FAMILY_STATUS,
        'NAME_INCOME_TYPE': data.NAME_INCOME_TYPE
    }])

    if data.model_type == "logistic":
        input_scaled = hc_scaler.transform(input_df)
        proba = to_python_float(hc_lr.predict_proba(input_scaled)[0][1])
    else:
        proba = to_python_float(hc_xgb.predict_proba(input_df)[0][1])

    risk_score = to_python_int(round(proba * 100))

    feature_importance = {
        'EXT_SOURCE_3': round(float(data.EXT_SOURCE_3) * 0.46, 4),
        'EXT_SOURCE_2': round(float(data.EXT_SOURCE_2) * 0.44, 4),
        'AGE_DAYS': round(float(data.AGE_DAYS) * 0.002, 2),
        'DAYS_EMPLOYED': round(float(data.DAYS_EMPLOYED) * 0.002, 2),
        'AMT_CREDIT': round(float(data.AMT_CREDIT) * 0.0001, 2)
    }

    return {
        "dataset": "Home Credit",
        "model_used": data.model_type,
        "risk_score": risk_score,
        "default_probability": round(proba * 100, 2),
        "risk_label": get_risk_label(risk_score),
        "decision": get_decision(risk_score),
        "top_factors": feature_importance
    }
