# %% [markdown]
# # Amazon employee reviews  
# 
# In this project, we will explore the idea of using employee reviews to determine  
# the probability of an employee leaving the company or not, so HR can act accordingly.  
# 
# The prediction of whether an employee leaves the company will be done using   
# logistic regression and the project will be carried out in different stages:  
# 1. Exploratory Data Analysis  
# 2. Model training, evaluation and selection  
# 3. Model deployment  
# 
# 
# 
# 

# %% [markdown]
# ## Dataset  
# The dataset was obtained from Kaggle and can be found here:  
# https://www.kaggle.com/datasets/nikhilraj7700/amazon-employee-reviews  
# 
# From the reference, the variable descriptions are:  
# Index: An exclusive identifier for each individual review entry.
# 
# * **Name:** The job title or role of the employee providing the review.  
# * **Place:** The geographical location or city where the employee works.  
# * **Job Type:** The employment status of the reviewer (e.g., Full Time).  
# * **Department:** The specific department or functional area within the organization.  
# * **Date:** The date when the review was submitted.  
# * **Overall Rating:** A numerical rating given by the employee for their overall job satisfaction.  
# * **Work Life Balance:** Rating indicating the work-life balance experienced by the employee.  
# * **Skill Development:** Rating reflecting the opportunities for skill enhancement and growth.  
# * **Salary and Benefits:** Rating assessing the satisfaction with compensation and benefits.  
# * **Job Security:** Rating expressing the employee's sense of job security.  
# * **Career Growth:** Rating indicating the perceived career advancement opportunities.  
# * **Work Satisfaction:** Rating showcasing the employee's contentment with their work.  
# * **Likes:** Positive aspects and pros highlighted by the employee in their review.  
# * **Dislikes:** Negative aspects and cons mentioned by the employee in their review.  

# %% [markdown]
# ## Exploratory Data Analysis

# %% [markdown]
# **Data importing**

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import pickle

# %%
original_df = pd.read_csv("./data/Amazon_Reviews.csv")

# %%
df = original_df.copy()

# %% [markdown]
# **Basic info**

# %% [markdown]
# We can see at a glance that:  
# + "Place" has null values
# + "Likes" and "Dislikes" are natural language, well suited for NLP analysis, which is not our case
# + "Date" is not of interest for this analysis
# + "Name", "Place", "Job_type", and "Department" are categorical variables, so we need to see how many occurrences are of each type
# + We will use "overall_rating" as the dependant variable, assuming that its score reflects the possibility of the employee leaving the company. Overall, this number reflects how happy is the employee, and the more *unhappy* the more likely to leave the company.  
# 
# In the next section, we will clean the dataset addressing null values, number of occurrences for categorical variables and establish consistency across names in the columns and the text content.

# %% [markdown]
# **Null values**

# %%
#fill null values with an "Unknown" category
df["Place"] = df["Place"].fillna("Unknown")

# %% [markdown]
# **Column names**

# %%
df.columns = [c.upper() for c in df.columns]

# %%
# keep selected columns
df.drop(columns = ["LIKES", "DISLIKES", "DATE"], inplace=True)
df.rename(columns = {"NAME":"POSITION"}, inplace=True)

# %% [markdown]
# **Categorical and non-categorical variables**

# %%

# %%
categorical_variables = []
numerical_variables = []
for c in df.columns:
    if df[c].dtype=="O":
        categorical_variables.append(c)
    else:
        numerical_variables.append(c)

# %% [markdown]
# **Unique values**

# %%
#  As noted before, "JOB_TYPE" has one unique value, so no point in using this as a variable that has an impact
df.drop(columns = ["JOB_TYPE"], inplace=True)

# %%
categorical_variables.remove("JOB_TYPE")

# %% [markdown]
# ## Target variable  
# We will use the overall rating as an indicator that the employee es happy or not

# %%
df["IS_HAPPY"] = (df["OVERALL_RATING"]>=3.0).astype(int)

# %%
# This is a slightly imbalanced class distribution, but we will see if this
# needs some adjustment to improve performance

# %% [markdown]
# ## Train-test split

# %%


# %%
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


# %%
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# %%
y_train = df_train.IS_HAPPY.values
y_val = df_val.IS_HAPPY.values
y_test = df_test.IS_HAPPY.values

df_train.drop(columns = ['OVERALL_RATING',"IS_HAPPY"], inplace=True)
df_val.drop(columns = ['OVERALL_RATING',"IS_HAPPY"], inplace=True)
df_test.drop(columns = ['OVERALL_RATING',"IS_HAPPY"], inplace=True)

# %%
numerical_variables.remove("OVERALL_RATING")


# %% [markdown]
# ## Using the model  
# We will first re-train the model but now using the complete full training set,   
# which would be 80% of the total dataset

# %% [markdown]
# ### Full train  (80%)

# %%
dicts_full_train = df_full_train[categorical_variables + numerical_variables].to_dict(orient='records')

# %%
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

# %%
model = LogisticRegression(solver='lbfgs')
model.fit(X_full_train, y_full_train)


# %% [markdown]
# ### **Save and load the model**  
# No regularization was done in this model


# %%
output_file = 'model.bin'
with open(output_file, "wb") as f_out:
    pickle.dump((dv,model), f_out)

print("Trained and save the model ", output_file)