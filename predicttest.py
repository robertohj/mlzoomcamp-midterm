import requests

test_employee ={
    "POSITION":                 "Software Development Engineer",
    "PLACE":                           "Hyderabad/Secunderabad",
    "DEPARTMENT":             "Software Development Department",
    "WORK_LIFE_BALANCE":                                    4.0,
    "SKILL_DEVELOPMENT":                                    5.0,
    "SALARY_AND_BENEFITS":                                  5.0,
    "JOB_SECURITY":                                         1.0,
    "CAREER_GROWTH":                                        5.0,
    "WORK_SATISFACTION":                                    3.0
}

url = "http://127.0.0.1:9696/predict"

response = requests.post(url,json=test_employee)
print(response.json())
