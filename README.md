# **Amazon employee reviews  - ML Zoomcamp Midterm Project**
## Problem description
In this project, we will use different aspects of the Amazon employee reviews to determine  
the probability of an employee leaving the company or not, so HR can act accordingly.  

The prediction of whether an employee leaves the company will be done using   
logistic regression and the project will be carried out in different stages:  
1. Exploratory Data Analysis  
2. Model training, evaluation and selection  
3. Model deployment and serving  


### Dataset  
The dataset was obtained from Kaggle and can be found here:  
https://www.kaggle.com/datasets/nikhilraj7700/amazon-employee-reviews  

From the reference, the variable descriptions are:  
Index: An exclusive identifier for each individual review entry.

* **Name:** The job title or role of the employee providing the review.  
* **Place:** The geographical location or city where the employee works.  
* **Job Type:** The employment status of the reviewer (e.g., Full Time).  
* **Department:** The specific department or functional area within the organization.  
* **Date:** The date when the review was submitted.  
* **Overall Rating:** A numerical rating given by the employee for their overall job satisfaction.  
* **Work Life Balance:** Rating indicating the work-life balance experienced by the employee.  
* **Skill Development:** Rating reflecting the opportunities for skill enhancement and growth.  
* **Salary and Benefits:** Rating assessing the satisfaction with compensation and benefits.  
* **Job Security:** Rating expressing the employee's sense of job security.  
* **Career Growth:** Rating indicating the perceived career advancement opportunities.  
* **Work Satisfaction:** Rating showcasing the employee's contentment with their work.  
* **Likes:** Positive aspects and pros highlighted by the employee in their review.  
* **Dislikes:** Negative aspects and cons mentioned by the employee in their review.  


## Jupyter Notebook  
The amazon-employee-review.ipynb file is the Notebook where EDA, training, evaluation and testing was done. More details can be seen inside. Hint: you can use the "OUTLINE" panel to easily navigate through the sections marked with MD language.  

## Training and Testing scripts  
The training and testing sections from the notebook were 
exported as separate .py scripts to be executed independently
and at will. Such files are:  
 + train_model_amazon_employee_review.py
 + predict_model_employee_review.py
 + predicttest.py  

## Model deployment  
The model was deployed using Flask in the file predict_amazon_employee_review.py  

## Environment  
A .venv virtual environment was used to isolate the packages used in this project. In that environment, the corresponding modules were installed using pip. The modules and their version numbers were dumped into the requirements.txt file.  

The environment can be recreated by using the following command:  
```python -m venv -r requirements.txt```  

## Production  
Waitress was used to build the model production-ready and serve. This package was also used to launch the API from its Docker container. 

## Containerization  
A container was used in Docker to isolate the software package and dependencies in the OS. The docker file Dockerfile is provided with the corresponding configuration that sets the production environment, installs dependencies and launches the Flask application to serve in production using Waitress.  

## Interaction  
In the notebook, the section *Serving the Model* provides evidence of the configuration, testing and production of the model. Due to time constrains, the Cloud deployment is left for future work. 







