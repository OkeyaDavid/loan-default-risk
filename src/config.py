"""Central settings: which columns are used, which are refused, and the loan economics.

Everything a reviewer would challenge lives here, in one file, with a written reason.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "application_train.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"
for _d in (PROCESSED, MODELS, DOCS):
    _d.mkdir(parents=True, exist_ok=True)

TARGET = "TARGET"            # 1 = client had payment difficulties (the dataset's definition of default)
RANDOM_STATE = 42
TEST_SIZE = 0.25             # held-out applicants, never seen during training or tuning

# --- Columns refused, with the rationale a credit-risk reviewer would ask for -------------
EXCLUDED = {
    "CODE_GENDER": (
        "Protected attribute. Sex is a prohibited ground under the Canadian Human Rights Act "
        "and provincial codes, and under US ECOA/Regulation B. Excluded from every model variant."
    ),
    "SK_ID_CURR": "Row identifier. Carries no information about the borrower.",
}

# Age is also a protected ground (ECOA, Canadian human rights codes), but unlike sex it is
# predictive of default and routinely used in scorecards. We train WITH and WITHOUT it and
# publish the AUC cost, so the choice is evidenced rather than asserted. See docs/model_card.md.
AGE_FEATURES = ["AGE_YEARS"]

# --- The ~25 features we keep, each one explainable to a credit officer -------------------
NUMERIC = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",   # external credit-bureau style scores
    "AMT_INCOME_TOTAL",        # annual income
    "AMT_CREDIT",              # loan amount
    "AMT_ANNUITY",             # annual payment
    "AMT_GOODS_PRICE",         # price of the financed good
    "DAYS_BIRTH",              # -> AGE_YEARS
    "DAYS_EMPLOYED",           # -> EMPLOYED_YEARS + IS_UNEMPLOYED
    "DAYS_ID_PUBLISH",         # -> YEARS_SINCE_ID_PUBLISH (identity-document stability)
    "DAYS_REGISTRATION",       # -> YEARS_SINCE_REGISTRATION
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "REGION_POPULATION_RELATIVE",
    "REGION_RATING_CLIENT_W_CITY",
    "DEF_30_CNT_SOCIAL_CIRCLE",   # defaults observed in the applicant's social circle
    "AMT_REQ_CREDIT_BUREAU_QRT",  # credit-bureau enquiries in the last quarter
]
CATEGORICAL = [
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]

# Engineered in features.py (affordability and stability, the ratios a credit officer uses)
ENGINEERED = [
    "AGE_YEARS", "EMPLOYED_YEARS", "IS_UNEMPLOYED",
    "YEARS_SINCE_REGISTRATION", "YEARS_SINCE_ID_PUBLISH",
    "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME", "CREDIT_TO_GOODS",
    "CREDIT_TERM_YEARS", "INCOME_PER_PERSON", "EXT_SOURCE_MEAN", "EXT_SOURCE_MIN",
]

# --- Loan economics for the cut-off analysis ---------------------------------------------
# Stated assumptions, not data. Shown on every chart and in the app.
LGD = 0.45       # loss given default: share of the balance lost when a loan defaults
MARGIN = 0.08    # lifetime margin earned on a loan that performs
# Break-even default probability p* solves (1-p)*MARGIN = p*LGD  ->  p* = MARGIN / (MARGIN + LGD)
BREAK_EVEN_PD = MARGIN / (MARGIN + LGD)   # ~0.151
