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


# ============================================
# INPUT SCHEMAS
# ============================================

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


# ============================================
# HELPER FUNCTIONS
# ============================================

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


# ============================================
# FEATURE 1 — INTEREST RATE PERSONALIZER
# Based on predicted default probability,
# we recommend a fair interest rate range
# that compensates the lender for the risk
# while being transparent to the borrower.
# ============================================

def get_interest_rate_recommendation(proba: float, dataset: str) -> dict:
    """
    Maps predicted default probability to a
    recommended interest rate range and tier.
    Thresholds derived from dataset statistics:
    - LC average rate: 13.2%, range 5.3%-30.9%
    - HC average annuity-to-credit ratio used
      as proxy for effective rate
    """
    pct = proba * 100

    if dataset == "lending_club":
        if pct < 15:
            rate_min, rate_max = 5.5, 9.9
            tier = "Prime"
            explanation = (
                "Excellent credit profile. Qualifies for "
                "our lowest rate tier reserved for the "
                "most creditworthy borrowers."
            )
        elif pct < 25:
            rate_min, rate_max = 10.0, 13.9
            tier = "Near-Prime"
            explanation = (
                "Good credit profile with low default risk. "
                "Eligible for competitive rates with standard "
                "loan terms."
            )
        elif pct < 40:
            rate_min, rate_max = 14.0, 19.9
            tier = "Standard"
            explanation = (
                "Moderate risk profile. Rate reflects elevated "
                "DTI or interest rate exposure. Improving DTI "
                "below 20% could unlock better rates."
            )
        elif pct < 60:
            rate_min, rate_max = 20.0, 25.9
            tier = "Subprime"
            explanation = (
                "Higher risk profile detected. Rate adjusted to "
                "compensate for default probability. Reducing "
                "revolving utilization could improve tier."
            )
        else:
            rate_min, rate_max = 26.0, 30.9
            tier = "High Risk"
            explanation = (
                "Risk profile exceeds standard lending thresholds. "
                "If approved, high rate applies. Recommend "
                "addressing key risk factors before reapplying."
            )

    else:  # home_credit
        if pct < 10:
            rate_min, rate_max = 8.0, 12.0
            tier = "Prime"
            explanation = (
                "Strong external credit scores and stable "
                "employment indicate excellent repayment capacity."
            )
        elif pct < 20:
            rate_min, rate_max = 12.1, 16.0
            tier = "Near-Prime"
            explanation = (
                "Good credit bureau scores with manageable "
                "credit-to-income ratio. Competitive rates apply."
            )
        elif pct < 40:
            rate_min, rate_max = 16.1, 22.0
            tier = "Standard"
            explanation = (
                "Moderate external credit scores detected. "
                "Improving EXT_SOURCE scores through timely "
                "repayments could lower rate significantly."
            )
        elif pct < 60:
            rate_min, rate_max = 22.1, 28.0
            tier = "Subprime"
            explanation = (
                "Lower credit bureau scores and income profile "
                "indicate elevated risk. Rate adjusted accordingly."
            )
        else:
            rate_min, rate_max = 28.1, 36.0
            tier = "High Risk"
            explanation = (
                "Credit profile significantly below thresholds. "
                "High rate required if loan proceeds. Recommend "
                "improving external credit scores first."
            )

    return {
        "recommended_rate_min": rate_min,
        "recommended_rate_max": rate_max,
        "rate_tier": tier,
        "rate_explanation": explanation
    }


# ============================================
# FEATURE 2 — AFFORDABILITY COACH
# Uses SHAP-derived feature thresholds from
# training data to identify exactly where the
# borrower falls short and what they can do.
# Thresholds derived from dataset analysis:
# LC: median/quartile values from 50k sample
# HC: median/quartile values from 50k sample
# ============================================

def get_affordability_coaching_lc(data: LendingClubInput, risk_score: int) -> dict:
    """
    Compares borrower inputs against safe thresholds
    derived from Lending Club training data analysis.
    Only generates suggestions for values outside
    safe ranges — not generic advice.
    """
    suggestions = []
    strengths = []
    improvement_potential = 0

    # --- Interest Rate (most important SHAP feature, weight 0.530) ---
    if data.int_rate > 20:
        suggestions.append({
            "factor": "Interest Rate",
            "current": f"{data.int_rate}%",
            "target": "Below 15%",
            "impact": "High",
            "advice": (
                f"Your interest rate of {data.int_rate}% is the strongest "
                "predictor of default in our model. Negotiating or "
                "refinancing to below 15% could reduce your risk score "
                "by approximately 15-20 points."
            )
        })
        improvement_potential += 20
    elif data.int_rate > 15:
        suggestions.append({
            "factor": "Interest Rate",
            "current": f"{data.int_rate}%",
            "target": "Below 12%",
            "impact": "Medium",
            "advice": (
                f"Your rate of {data.int_rate}% is moderate. "
                "Improving credit score to qualify for sub-12% "
                "rates would meaningfully lower your risk profile."
            )
        })
        improvement_potential += 10
    else:
        strengths.append("Interest rate is within a healthy range.")

    # --- DTI Ratio (SHAP weight 0.159) ---
    if data.dti > 35:
        suggestions.append({
            "factor": "Debt-to-Income Ratio",
            "current": f"{data.dti}%",
            "target": "Below 20%",
            "impact": "High",
            "advice": (
                f"Your DTI of {data.dti}% is significantly above the "
                "safe threshold of 20%. Paying down existing debts "
                "or increasing income would have the second largest "
                "impact on your loan eligibility."
            )
        })
        improvement_potential += 18
    elif data.dti > 20:
        suggestions.append({
            "factor": "Debt-to-Income Ratio",
            "current": f"{data.dti}%",
            "target": "Below 20%",
            "impact": "Medium",
            "advice": (
                f"DTI of {data.dti}% is slightly elevated. "
                "Reducing monthly debt obligations by 10-15% "
                "would bring this within the safe range."
            )
        })
        improvement_potential += 8
    else:
        strengths.append("Debt-to-income ratio is healthy.")

    # --- Revolving Utilization (credit usage) ---
    if data.revol_util > 75:
        suggestions.append({
            "factor": "Credit Utilization",
            "current": f"{data.revol_util}%",
            "target": "Below 30%",
            "impact": "Medium",
            "advice": (
                f"You are using {data.revol_util}% of available revolving "
                "credit. Lenders prefer below 30%. Paying down credit "
                "card balances is the fastest way to improve this."
            )
        })
        improvement_potential += 10
    elif data.revol_util > 30:
        suggestions.append({
            "factor": "Credit Utilization",
            "current": f"{data.revol_util}%",
            "target": "Below 30%",
            "impact": "Low",
            "advice": (
                f"Utilization at {data.revol_util}% is slightly high. "
                "Bringing it below 30% would positively "
                "impact your credit profile."
            )
        })
        improvement_potential += 5
    else:
        strengths.append("Credit utilization is excellent.")

    # --- Annual Income ---
    if data.annual_inc < 40000:
        suggestions.append({
            "factor": "Annual Income",
            "current": f"${data.annual_inc:,.0f}",
            "target": "Above $50,000",
            "impact": "Medium",
            "advice": (
                "Income below $40,000 limits borrowing capacity "
                "relative to loan size. Consider a smaller loan "
                "amount or a co-borrower to strengthen the application."
            )
        })
        improvement_potential += 8
    else:
        strengths.append("Income level supports loan repayment capacity.")

    # --- Loan Term ---
    if data.term == 60:
        suggestions.append({
            "factor": "Loan Term",
            "current": "60 months",
            "target": "36 months",
            "impact": "Medium",
            "advice": (
                "60-month loans carry higher default risk in our model "
                "due to prolonged repayment exposure. If monthly "
                "payments allow, switching to 36 months reduces "
                "risk score by approximately 8-12 points."
            )
        })
        improvement_potential += 10
    else:
        strengths.append("36-month term is lower risk than 60-month.")

    # --- Employment Length ---
    if data.emp_length < 2:
        suggestions.append({
            "factor": "Employment Stability",
            "current": f"{data.emp_length} years",
            "target": "2+ years",
            "impact": "Low",
            "advice": (
                "Less than 2 years at current employer is a mild "
                "risk signal. Applying after establishing longer "
                "employment history would strengthen the profile."
            )
        })
        improvement_potential += 5
    else:
        strengths.append("Employment length demonstrates stability.")

    # Summary message
    if risk_score < 30:
        summary = (
            "Your profile is strong. You qualify for loan approval "
            "with favorable terms."
        )
    elif risk_score < 60:
        summary = (
            f"Your profile has potential but needs improvement in "
            f"{len(suggestions)} area(s). Addressing the High impact "
            f"factors could reduce your risk score by approximately "
            f"{min(improvement_potential, 35)} points."
        )
    else:
        summary = (
            f"Your current profile presents significant default risk. "
            f"Addressing all flagged areas could improve your score "
            f"by an estimated {min(improvement_potential, 45)} points. "
            f"We recommend reapplying after making these improvements."
        )

    return {
        "summary": summary,
        "improvement_potential_points": min(improvement_potential, 45),
        "suggestions": suggestions,
        "strengths": strengths
    }


def get_affordability_coaching_hc(data: HomeCreditInput, risk_score: int) -> dict:
    """
    Compares Home Credit applicant inputs against
    safe thresholds derived from training data.
    EXT_SOURCE scores are the dominant SHAP features
    (combined weight > 0.90) so they receive priority.
    """
    suggestions = []
    strengths = []
    improvement_potential = 0

    # --- EXT_SOURCE_3 (SHAP weight 0.460 — strongest predictor) ---
    if data.EXT_SOURCE_3 < 0.3:
        suggestions.append({
            "factor": "External Credit Score 3",
            "current": f"{data.EXT_SOURCE_3:.2f}",
            "target": "Above 0.5",
            "impact": "High",
            "advice": (
                f"EXT_SOURCE_3 of {data.EXT_SOURCE_3:.2f} is significantly "
                "below the safe threshold. This is the single most "
                "influential factor in our model. Improving credit bureau "
                "scores through timely repayments and reducing outstanding "
                "debts is the most impactful action you can take."
            )
        })
        improvement_potential += 25
    elif data.EXT_SOURCE_3 < 0.5:
        suggestions.append({
            "factor": "External Credit Score 3",
            "current": f"{data.EXT_SOURCE_3:.2f}",
            "target": "Above 0.6",
            "impact": "High",
            "advice": (
                f"EXT_SOURCE_3 of {data.EXT_SOURCE_3:.2f} is below the "
                "optimal threshold. Consistent on-time payments over "
                "6-12 months typically improves this score meaningfully."
            )
        })
        improvement_potential += 15
    else:
        strengths.append("External Credit Score 3 is within a healthy range.")

    # --- EXT_SOURCE_2 (SHAP weight 0.438) ---
    if data.EXT_SOURCE_2 < 0.3:
        suggestions.append({
            "factor": "External Credit Score 2",
            "current": f"{data.EXT_SOURCE_2:.2f}",
            "target": "Above 0.5",
            "impact": "High",
            "advice": (
                f"EXT_SOURCE_2 of {data.EXT_SOURCE_2:.2f} is a critical "
                "risk signal. This score reflects your credit bureau "
                "history. Clearing any outstanding defaults or "
                "delinquencies would have immediate positive impact."
            )
        })
        improvement_potential += 22
    elif data.EXT_SOURCE_2 < 0.5:
        suggestions.append({
            "factor": "External Credit Score 2",
            "current": f"{data.EXT_SOURCE_2:.2f}",
            "target": "Above 0.6",
            "impact": "High",
            "advice": (
                f"EXT_SOURCE_2 at {data.EXT_SOURCE_2:.2f} needs improvement. "
                "Avoiding new credit applications and maintaining "
                "existing accounts in good standing will help."
            )
        })
        improvement_potential += 12
    else:
        strengths.append("External Credit Score 2 is healthy.")

    # --- Age (older = more stable, SHAP weight 0.228) ---
    age_years = data.AGE_DAYS / 365
    if age_years < 25:
        suggestions.append({
            "factor": "Applicant Age",
            "current": f"{age_years:.0f} years",
            "target": "25+ years",
            "impact": "Low",
            "advice": (
                "Younger applicants statistically show higher default "
                "rates. Building a stronger credit history and "
                "savings record over the next few years will "
                "significantly improve future applications."
            )
        })
        improvement_potential += 5
    else:
        strengths.append("Age profile indicates financial maturity.")

    # --- Employment (SHAP weight 0.212) ---
    emp_years = data.DAYS_EMPLOYED / 365
    if emp_years < 1:
        suggestions.append({
            "factor": "Employment Duration",
            "current": f"{emp_years:.1f} years",
            "target": "2+ years",
            "impact": "Medium",
            "advice": (
                "Less than 1 year of employment is a notable risk "
                "factor. Stable employment for 2+ years at the "
                "same employer significantly improves loan eligibility."
            )
        })
        improvement_potential += 10
    elif emp_years < 2:
        suggestions.append({
            "factor": "Employment Duration",
            "current": f"{emp_years:.1f} years",
            "target": "2+ years",
            "impact": "Low",
            "advice": (
                f"{emp_years:.1f} years of employment is slightly below "
                "the preferred threshold. Maintaining current "
                "employment for another year would help."
            )
        })
        improvement_potential += 5
    else:
        strengths.append("Employment duration indicates income stability.")

    # --- Credit-to-Income Ratio ---
    if data.AMT_INCOME_TOTAL > 0:
        credit_to_income = data.AMT_CREDIT / data.AMT_INCOME_TOTAL
        if credit_to_income > 5:
            suggestions.append({
                "factor": "Credit-to-Income Ratio",
                "current": f"{credit_to_income:.1f}x annual income",
                "target": "Below 4x annual income",
                "impact": "Medium",
                "advice": (
                    f"Requested credit is {credit_to_income:.1f}x your annual "
                    "income, which is above comfortable lending thresholds. "
                    "Requesting a smaller credit amount or increasing "
                    "income would improve this ratio."
                )
            })
            improvement_potential += 8
        else:
            strengths.append("Credit amount is proportionate to income.")

    # --- Property ownership as stability signal ---
    if data.FLAG_OWN_REALTY == 1:
        strengths.append("Property ownership demonstrates financial stability.")

    # Summary
    if risk_score < 30:
        summary = (
            "Strong applicant profile. External credit scores and "
            "employment history support loan approval."
        )
    elif risk_score < 60:
        summary = (
            f"Profile shows promise but has {len(suggestions)} area(s) "
            f"requiring attention. Improving the High impact factors "
            f"could reduce risk score by approximately "
            f"{min(improvement_potential, 35)} points."
        )
    else:
        summary = (
            f"Current profile presents elevated default risk. "
            f"Focus on improving External Credit Scores as the "
            f"highest priority action. Addressing all flagged areas "
            f"could improve your score by an estimated "
            f"{min(improvement_potential, 50)} points."
        )

    return {
        "summary": summary,
        "improvement_potential_points": min(improvement_potential, 50),
        "suggestions": suggestions,
        "strengths": strengths
    }


# ============================================
# ROUTES
# ============================================

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

    # ── New Features ──
    interest_rate_rec = get_interest_rate_recommendation(proba, "lending_club")
    affordability_coach = get_affordability_coaching_lc(data, risk_score)

    return {
        "dataset": "Lending Club",
        "model_used": data.model_type,
        "risk_score": risk_score,
        "default_probability": round(proba * 100, 2),
        "risk_label": get_risk_label(risk_score),
        "decision": get_decision(risk_score),
        "top_factors": feature_importance,
        "interest_rate_recommendation": interest_rate_rec,
        "affordability_coach": affordability_coach
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

    # ── New Features ──
    interest_rate_rec = get_interest_rate_recommendation(proba, "home_credit")
    affordability_coach = get_affordability_coaching_hc(data, risk_score)

    return {
        "dataset": "Home Credit",
        "model_used": data.model_type,
        "risk_score": risk_score,
        "default_probability": round(proba * 100, 2),
        "risk_label": get_risk_label(risk_score),
        "decision": get_decision(risk_score),
        "top_factors": feature_importance,
        "interest_rate_recommendation": interest_rate_rec,
        "affordability_coach": affordability_coach
    }
