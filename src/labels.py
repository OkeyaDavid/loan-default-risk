"""Plain-English names for model features. Deliberately dependency-free: the Streamlit
app imports this, and must not pull in matplotlib or SHAP just to label a column.
"""
from __future__ import annotations

# Plain-English wording for the features a reviewer will actually see.
LABELS = {
    "EXT_SOURCE_1": "external credit score 1",
    "EXT_SOURCE_2": "external credit score 2",
    "EXT_SOURCE_3": "external credit score 3",
    "EXT_SOURCE_MEAN": "average external credit score",
    "EXT_SOURCE_MIN": "weakest external credit score",
    "CREDIT_TO_INCOME": "loan size relative to income",
    "ANNUITY_TO_INCOME": "annual payment relative to income",
    "CREDIT_TO_GOODS": "loan relative to the price of the purchase",
    "CREDIT_TERM_YEARS": "repayment term",
    "INCOME_PER_PERSON": "income per household member",
    "AMT_INCOME_TOTAL": "annual income",
    "AMT_CREDIT": "loan amount",
    "AMT_ANNUITY": "annual payment",
    "AMT_GOODS_PRICE": "price of the purchase",
    "AGE_YEARS": "age",
    "EMPLOYED_YEARS": "years in current employment",
    "IS_UNEMPLOYED": "not currently employed",
    "YEARS_SINCE_REGISTRATION": "years since registration record",
    "YEARS_SINCE_ID_PUBLISH": "years since identity document issued",
    "CNT_CHILDREN": "number of children",
    "CNT_FAM_MEMBERS": "household size",
    "REGION_POPULATION_RELATIVE": "how urban the region is",
    "REGION_RATING_CLIENT_W_CITY": "region rating",
    "DEF_30_CNT_SOCIAL_CIRCLE": "defaults in the applicant's social circle",
    "AMT_REQ_CREDIT_BUREAU_QRT": "credit-bureau enquiries this quarter",
    "NAME_CONTRACT_TYPE": "loan type",
    "NAME_INCOME_TYPE": "income type",
    "NAME_EDUCATION_TYPE": "education",
    "NAME_FAMILY_STATUS": "family status",
    "NAME_HOUSING_TYPE": "housing",
    "OCCUPATION_TYPE": "occupation",
    "ORGANIZATION_TYPE": "employer type",
    "FLAG_OWN_CAR": "owns a car",
    "FLAG_OWN_REALTY": "owns property",
}


def pretty(raw_name: str) -> str:
    """Turn a transformed column name ('cat__OCCUPATION_TYPE_Laborers') into words."""
    name = raw_name.split("__", 1)[-1]
    for col, label in LABELS.items():
        if name == col:
            return label
        if name.startswith(col + "_"):
            value = name[len(col) + 1:].replace("_", " ")
            return f"{label} = {value}"
    return name.replace("_", " ").lower()
