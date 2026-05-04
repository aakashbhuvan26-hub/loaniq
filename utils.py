"""
backend/utils.py
Utility functions for LoanIQ — data validation, EMI calculation,
feature engineering, and report generation.
"""

import math
from typing import Any


# ─── Validation ──────────────────────────────────────────────────────────────

FIELD_RULES = {
    "gender":             {"type": int,   "choices": [0, 1]},
    "married":            {"type": int,   "choices": [0, 1]},
    "dependents":         {"type": int,   "choices": [0, 1, 2, 3]},
    "education":          {"type": int,   "choices": [0, 1]},
    "self_employed":      {"type": int,   "choices": [0, 1]},
    "applicant_income":   {"type": float, "min": 0},
    "coapplicant_income": {"type": float, "min": 0},
    "loan_amount":        {"type": float, "min": 1},
    "loan_term":          {"type": float, "min": 1},
    "credit_history":     {"type": int,   "choices": [0, 1]},
    "property_area":      {"type": int,   "choices": [0, 1, 2]},
}


def validate_input(data: dict) -> tuple[bool, str]:
    """
    Validate incoming prediction request.
    Returns (is_valid: bool, error_message: str).
    """
    for field, rules in FIELD_RULES.items():
        if field not in data:
            return False, f"Missing required field: '{field}'"

        try:
            val = rules["type"](data[field])
        except (ValueError, TypeError):
            return False, f"Field '{field}' must be of type {rules['type'].__name__}"

        if "choices" in rules and val not in rules["choices"]:
            return False, f"Field '{field}' must be one of {rules['choices']}, got {val}"

        if "min" in rules and val < rules["min"]:
            return False, f"Field '{field}' must be >= {rules['min']}, got {val}"

    return True, ""


# ─── Financial Calculations ───────────────────────────────────────────────────

def calc_emi(loan_amount_k: float, annual_rate_pct: float, term_months: int) -> float:
    """
    Calculate monthly EMI using the standard amortisation formula.

    Args:
        loan_amount_k:  Loan amount in ₹ thousands
        annual_rate_pct: Annual interest rate as a percentage (e.g. 8.5)
        term_months:    Loan tenure in months

    Returns:
        Monthly EMI in ₹
    """
    principal = loan_amount_k * 1000
    if annual_rate_pct == 0:
        return principal / term_months if term_months > 0 else 0.0
    r = annual_rate_pct / 100 / 12
    emi = principal * r * (1 + r) ** term_months / ((1 + r) ** term_months - 1)
    return round(emi, 2)


def calc_dti(emi: float, total_income: float) -> float:
    """Debt-to-income ratio as a percentage."""
    if total_income <= 0:
        return 100.0
    return round(emi / total_income * 100, 2)


def loan_summary(
    income: float,
    co_income: float,
    loan_amount_k: float,
    term_months: int,
    annual_rate_pct: float = 8.0,
) -> dict[str, Any]:
    """Return a full financial summary dict for a loan application."""
    total_income = income + co_income
    emi = calc_emi(loan_amount_k, annual_rate_pct, term_months)
    total_payable = round(emi * term_months, 2)
    total_interest = round(total_payable - loan_amount_k * 1000, 2)
    dti = calc_dti(emi, total_income)

    return {
        "total_income":   total_income,
        "emi":            emi,
        "total_payable":  total_payable,
        "total_interest": total_interest,
        "dti_ratio":      dti,
        "interest_rate":  annual_rate_pct,
        "term_months":    term_months,
        "loan_amount":    loan_amount_k * 1000,
    }


# ─── Feature Engineering ─────────────────────────────────────────────────────

def build_feature_vector(data: dict) -> list[float]:
    """
    Convert raw API payload to the ordered feature vector expected by the model.
    Feature order must match the training pipeline exactly.
    """
    return [
        float(data["gender"]),
        float(data["married"]),
        float(data["dependents"]),
        float(data["education"]),
        float(data["self_employed"]),
        float(data["applicant_income"]),
        float(data["coapplicant_income"]),
        float(data["loan_amount"]),
        float(data["loan_term"]),
        float(data["credit_history"]),
        float(data["property_area"]),
    ]


# ─── Label Maps ──────────────────────────────────────────────────────────────

AREA_LABELS = {0: "Rural", 1: "Semiurban", 2: "Urban"}


def humanize(data: dict) -> dict[str, str]:
    """Convert raw encoded values to human-readable labels for reports."""
    return {
        "Gender":            "Male" if data.get("gender") == 1 else "Female",
        "Marital Status":    "Married" if data.get("married") == 1 else "Unmarried",
        "Dependents":        str(data.get("dependents", 0)),
        "Education":         "Graduate" if data.get("education") == 1 else "Not Graduate",
        "Self Employed":     "Yes" if data.get("self_employed") == 1 else "No",
        "Applicant Income":  f"₹{data.get('applicant_income', 0):,.0f}/mo",
        "Co-applicant Income": f"₹{data.get('coapplicant_income', 0):,.0f}/mo",
        "Loan Amount":       f"₹{data.get('loan_amount', 0):.0f}K",
        "Loan Term":         f"{data.get('loan_term', 0):.0f} months",
        "Credit History":    "Good" if data.get("credit_history") == 1 else "Bad",
        "Property Area":     AREA_LABELS.get(data.get("property_area", 0), "Unknown"),
    }
