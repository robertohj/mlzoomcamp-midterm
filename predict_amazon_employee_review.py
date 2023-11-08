
# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import pickle

# load the model
output_file = 'model.bin'
with open(output_file, "rb") as f_in:
    (dv,model) = pickle.load(f_in)


# %%
employee ={
    "POSITION":                 "Software Development Engineer",
    "PLACE":                           "Hyderabad/Secunderabad",
    "DEPARTMENT":             "Software Development Department",
    "WORK_LIFE_BALANCE":                                  1.0,
    "SKILL_DEVELOPMENT":                                  2.0,
    "SALARY_AND_BENEFITS":                                5.0,
    "JOB_SECURITY":                                       1.0,
    "CAREER_GROWTH":                                      3.0,
    "WORK_SATISFACTION":                                  3.0,
}

# %%
X = dv.transform([employee])
proba = model.predict_proba(X)[0][1]

print("INPUT \n", employee, "\n")
print("Probability of leaving: ",proba)
