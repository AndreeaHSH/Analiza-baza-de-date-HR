import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Imports for regression and feature selection
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Loading and Data Preprocessing

url = "https://raw.githubusercontent.com/catamaican/AED/main/HRDataset_v14.csv"

# Read the dataset
all_data = pd.read_csv(url)

# Display some basic information
print("First rows of the dataset:")
print(all_data.head())
print("\nGeneral information about the dataset:")
all_data.info()
print(f"\nDataset size: {all_data.shape[0]} rows and {all_data.shape[1]} columns.\n")

# Handling missing values
all_data['DateofTermination'].fillna('Still Employed', inplace=True)
all_data['ManagerID'].fillna(0, inplace=True)

# Converting date columns
all_data['DOB'] = pd.to_datetime(all_data['DOB'], errors='coerce')
all_data['DateofHire'] = pd.to_datetime(all_data['DateofHire'], errors='coerce')
# For DateofTermination, conversion is attempted; if not possible (e.g., "Still Employed") it will result in NaT
all_data['DateofTermination_converted'] = pd.to_datetime(all_data['DateofTermination'], errors='coerce')

# Calculating age and company tenure
today = pd.Timestamp('today')
all_data['Age'] = ((today - all_data['DOB']).dt.days / 365).astype(int)
all_data['Experience'] = ((today - all_data['DateofHire']).dt.days / 365).astype(int)

# Mapping for performance score
performance_map = {'Exceeds': 3, 'Meets': 2, 'Needs Improvement': 1, 'PIP': 0}
all_data['PerformanceNumeric'] = all_data['PerformanceScore'].map(performance_map)

# Turnover indicator: 0 = still employed, 1 = left the company
all_data['Turnover'] = np.where(all_data['DateofTermination'] == 'Still Employed', 0, 1)

# For some analyses, it is desired that gender remains textual;
# if a numeric representation is needed, an additional column can be created.
all_data['Sex_numeric'] = all_data['Sex'].apply(lambda x: 1 if x == 'M' else 0)


# 2. Additional Transformations: Dummy Encoding

# Perform dummy encoding for the selected categorical variables
dummy_cols = ["MaritalDesc", "Department", "Position", "CitizenDesc", "RaceDesc", "EmploymentStatus", "PerformanceScore"]
df_dummies = pd.get_dummies(all_data, columns=dummy_cols, drop_first=True)

# Mapping for binary columns from dummy (e.g., HispanicLatino and Sex)
df_dummies['HispanicLatino'] = df_dummies['HispanicLatino'].apply(lambda x: 1 if x == 'Yes' else 0)
df_dummies['Sex'] = df_dummies['Sex'].apply(lambda x: 1 if x == 'M' else 0)

# Concatenate the original data with the dummy-encoded data
df_concat = pd.concat([all_data, df_dummies], axis=1)
# Remove duplicate columns
df_concat = df_concat.loc[:, ~df_concat.columns.duplicated()]
# If desired, remove the original columns that were dummy-encoded
df_concat = df_concat.drop(columns=dummy_cols, errors="ignore")

# 3. Descriptive Statistics and Initial Visualizations

# Display descriptive statistics for df_concat
print("\nDescriptive statistics (df_concat):")
print(df_concat.info())
print(df_concat.describe())

numeric_columns_all_data = all_data.select_dtypes(include=[np.number])
median_values = numeric_columns_all_data.median()
print(f"Salary - Median: {median_values['Salary']:.2f}")
print(f"Employee Satisfaction - Median: {median_values['EmpSatisfaction']:.2f}")

# Select numeric columns and calculate quartiles
numeric_columns = df_concat.select_dtypes(include=[np.number])
quartiles = numeric_columns.quantile([0.25, 0.5, 0.75])
print("\nQuartiles:\n", quartiles)
if 'Salary' in numeric_columns.columns:
    print(f"Salary - 25th Percentile: {quartiles.loc[0.25, 'Salary']:.2f}")
    print(f"Salary - 50th Percentile: {quartiles.loc[0.5, 'Salary']:.2f}")
    print(f"Salary - 75th Percentile: {quartiles.loc[0.75, 'Salary']:.2f}")
else:
    print("Column 'Salary' does not exist in the dataset.")
if 'EmpSatisfaction' in numeric_columns.columns:
    print(f"Employee Satisfaction - 25th Percentile: {quartiles.loc[0.25, 'EmpSatisfaction']:.2f}")
    print(f"Employee Satisfaction - 50th Percentile: {quartiles.loc[0.5, 'EmpSatisfaction']:.2f}")
    print(f"Employee Satisfaction - 75th Percentile: {quartiles.loc[0.75, 'EmpSatisfaction']:.2f}")
else:
    print("Column 'EmpSatisfaction' does not exist in the dataset.")

# Visualizing the salary distribution
sns.histplot(all_data['Salary'], kde=True)
plt.title("Salary Distribution")
plt.show()

sns.boxplot(x=all_data["Department"], y=all_data["Salary"])
plt.title("Salary Distribution by Department")
plt.xlabel("Department")
plt.ylabel("Salary")
plt.xticks(rotation=45)
plt.show()
# The histogram with KDE shows how salaries are distributed – usually, the distribution may be skewed, indicating the presence of extremes (very high salaries compared to the majority).

# Relationship between absences and salary
plt.scatter(all_data["Absences"], all_data["Salary"], alpha=0.5)
plt.title("Relationship between Absences and Salary")
plt.xlabel("Number of Absences")
plt.ylabel("Salary")
plt.show()

# Calculate Pearson correlation between absences and salary
corr, p_value = pearsonr(all_data["Absences"].dropna(), all_data["Salary"].dropna())
print(f"Pearson Correlation Coefficient: {corr:.2f}")
print(f"P-value: {p_value:.4f}")
# The scatter plot suggests a relationship between the number of absences and salary.
# The Pearson correlation (value and p-value) provides a quantitative measure: a coefficient closer to 0 indicates a weak relationship, while a value significantly different from 0 (and a low p-value) would suggest a stronger connection.

# Main recruitment source
recruitment_source_counts = all_data["RecruitmentSource"].value_counts()
print("Number of employees recruited by source:\n", recruitment_source_counts)
sns.barplot(x=recruitment_source_counts.index, y=recruitment_source_counts.values, ci=None)
plt.title("Distribution of Recruitment Sources")
plt.xlabel("Recruitment Source")
plt.ylabel("Number of Employees")
plt.xticks(rotation=45)
plt.show()

# Top 5 employees by salary
if {"Employee_Name", "Position", "Salary"}.issubset(all_data.columns):
    top_employees = all_data[["Employee_Name", "Position", "Salary"]].sort_values(by="Salary", ascending=False).head(5)
    print("Top 5 employees with the highest salaries:")
    print(top_employees)

# 4. Analysis and Visualization Functions

def plot_demographics(data):
    """
    Demographic distribution: age and gender within each department and position.
    """
    plt.figure(figsize=(12,6))
    sns.boxplot(data=data, x='Department', y='Age', hue='Sex')
    plt.title("Age Distribution by Department (colored by gender)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,6))
    sns.boxplot(data=data, x='Position', y='Age', hue='Sex')
    plt.title("Age Distribution by Position (colored by gender)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_salary_performance(data):
    """
    Salary analysis: average salary by department, relationship between salary vs. experience and
    salary vs. performance score; plus analysis by experience categories.
    """
    dept_salary = data.groupby("Department")["Salary"].mean().reset_index()
    plt.figure(figsize=(10,6))
    sns.barplot(data=dept_salary, x="Department", y="Salary")
    plt.title("Average Salary by Department")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="Experience", y="Salary", hue="Department")
    plt.title("Correlation between Experience and Salary")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="PerformanceNumeric", y="Salary", hue="Department")
    plt.title("Correlation between Performance Score and Salary")
    plt.tight_layout()
    plt.show()
    
    data['ExperienceCat'] = pd.cut(data['Experience'], bins=[0,5,10,20,50], labels=["0-5","5-10","10-20","20+"])
    plt.figure(figsize=(8,6))
    sns.boxplot(data=data, x="ExperienceCat", y="Salary")
    plt.title("Salary Distribution by Experience Categories")
    plt.tight_layout()
    plt.show()

def plot_career_progression(data):
    """
    Career progression: relationship between tenure and salary, differentiated by position and turnover indicator.
    """
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="Experience", y="Salary", hue="Position", style="Turnover")
    plt.title("Salary vs. Tenure (indicative of career progression)")
    plt.tight_layout()
    plt.show()

def plot_training_development(data):
    """
    Training and development analysis: distribution of the number of special projects (proxy for training)
    and its relationship with employee satisfaction.
    """
    if "SpecialProjectsCount" in data.columns:
        plt.figure(figsize=(8,6))
        sns.histplot(data["SpecialProjectsCount"], kde=True, bins=20)
        plt.title("Distribution of Special Projects Count")
        plt.tight_layout()
        plt.show()
    
        plt.figure(figsize=(8,6))
        sns.boxplot(data=data, x="SpecialProjectsCount", y="EmpSatisfaction")
        plt.title("Employee Satisfaction vs. Number of Special Projects")
        plt.tight_layout()
        plt.show()
    else:
        print("Column 'SpecialProjectsCount' is not available in the dataset.")

def analyze_turnover(data):
    """
    Turnover analysis: average turnover rate by department.
    """
    dept_turnover = data.groupby("Department")["Turnover"].mean().reset_index()
    plt.figure(figsize=(10,6))
    sns.barplot(data=dept_turnover, x="Department", y="Turnover")
    plt.title("Turnover Rate by Department")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_recruitment(data):
    """
    Recruitment source analysis: numeric performance and turnover rate based on recruitment source.
    """
    if "RecruitmentSource" in data.columns:
        plt.figure(figsize=(12,6))
        sns.boxplot(data=data, x="RecruitmentSource", y="PerformanceNumeric")
        plt.title("Performance by Recruitment Source")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        rec_turnover = data.groupby("RecruitmentSource")["Turnover"].mean().reset_index()
        plt.figure(figsize=(12,6))
        sns.barplot(data=rec_turnover, x="RecruitmentSource", y="Turnover")
        plt.title("Turnover Rate by Recruitment Source")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Column 'RecruitmentSource' is not available.")

def analyze_satisfaction(data):
    """
    Relationship between employee satisfaction and performance and absence indicators.
    Also calculates Pearson correlation coefficients.
    """
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="PerformanceNumeric", y="EmpSatisfaction", hue="Department")
    plt.title("Employee Satisfaction vs. Performance Score")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="Absences", y="EmpSatisfaction", hue="Department")
    plt.title("Employee Satisfaction vs. Absences")
    plt.tight_layout()
    plt.show()
    
    if data["PerformanceNumeric"].notnull().sum() > 0 and data["EmpSatisfaction"].notnull().sum() > 0:
        corr_perf, _ = pearsonr(data["PerformanceNumeric"].dropna(), data["EmpSatisfaction"].dropna())
        print(f"Correlation between Satisfaction and Performance: {corr_perf:.2f}")
    if data["Absences"].notnull().sum() > 0 and data["EmpSatisfaction"].notnull().sum() > 0:
        corr_abs, _ = pearsonr(data["Absences"].dropna(), data["EmpSatisfaction"].dropna())
        print(f"Correlation between Satisfaction and Absences: {corr_abs:.2f}")

def analyze_contracts(data):
    """
    Contract structure analysis: distribution of contract types and their impact on performance and turnover.
    """
    if "EmploymentStatus" in data.columns:
        status_counts = data["EmploymentStatus"].value_counts().reset_index()
        status_counts.columns = ["EmploymentStatus", "Count"]
        plt.figure(figsize=(8,6))
        sns.barplot(data=status_counts, x="EmploymentStatus", y="Count")
        plt.title("Distribution of Contract Types")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(12,6))
        sns.boxplot(data=data, x="EmploymentStatus", y="PerformanceNumeric")
        plt.title("Performance by Contract Type")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        status_turnover = data.groupby("EmploymentStatus")["Turnover"].mean().reset_index()
        plt.figure(figsize=(8,6))
        sns.barplot(data=status_turnover, x="EmploymentStatus", y="Turnover")
        plt.title("Turnover Rate by Contract Type")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Column 'EmploymentStatus' is not available.")

def analyze_salary_gender(data):
    """
    Analysis of salary differences by gender: salary distribution and average salary by gender.
    """
    plt.figure(figsize=(8,6))
    sns.boxplot(data=data, x="Sex", y="Salary")
    plt.title("Salary Distribution by Gender")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12,6))
    sns.boxplot(data=data, x="Department", y="Salary", hue="Sex")
    plt.title("Salary by Department, Differentiated by Gender")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    mean_salary = data.groupby("Sex")["Salary"].mean()
    print("Average salary by gender:")
    print(mean_salary)

def analyze_performance_salary_gender(data):
    """
    Correlation between performance and salary, analyzed separately by gender.
    """
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="PerformanceNumeric", y="Salary", hue="Sex")
    plt.title("Correlation between Performance and Salary (by Gender)")
    plt.tight_layout()
    plt.show()
    
    for gender in data['Sex'].unique():
        subset = data[data['Sex'] == gender]
        if subset["PerformanceNumeric"].notnull().sum() > 0 and subset["Salary"].notnull().sum() > 0:
            corr, _ = pearsonr(subset["PerformanceNumeric"].dropna(), subset["Salary"].dropna())
            print(f"Correlation between Performance and Salary for gender {gender}: {corr:.2f}")

def analyze_external_factors(data):
    """
    Impact of external factors: the influence of experience (and, if available, special projects)
    on salary, differentiated by gender.
    """
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="Experience", y="Salary", hue="Sex")
    plt.title("Impact of Experience on Salary (by Gender)")
    plt.tight_layout()
    plt.show()
    
    if "SpecialProjectsCount" in data.columns:
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=data, x="SpecialProjectsCount", y="Salary", hue="Sex")
        plt.title("Impact of Special Projects on Salary (by Gender)")
        plt.tight_layout()
        plt.show()

def analyze_career_progression_gender(data):
    """
    Career progression differentiated by gender: the relationship between tenure and salary for males and females.
    """
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="Experience", y="Salary", hue="Sex")
    plt.title("Career Progression: Salary vs. Tenure (by Gender)")
    plt.tight_layout()
    plt.show()

def analyze_remuneration_policy(data):
    """
    Evaluation of remuneration policies: how salaries are distributed based on performance score and gender,
    as well as the relationship between salary and employee satisfaction.
    """
    plt.figure(figsize=(12,6))
    sns.boxplot(data=data, x="PerformanceScore", y="Salary", hue="Sex")
    plt.title("Salary Distribution by Performance Score and Gender")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="EmpSatisfaction", y="Salary", hue="Sex")
    plt.title("Salary vs. Employee Satisfaction (by Gender)")
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(data, title="Correlation Matrix", drop_cols=None, figsize=(12, 10), cmap="coolwarm"):
    """
    Calculates and displays the correlation matrix for numeric variables in a DataFrame.
    Parameters:
      data (pd.DataFrame): The DataFrame for which correlations are calculated.
      title (str): The title of the plot.
      drop_cols (list, optional): List of columns to drop before calculation.
      figsize (tuple): Figure size.
      cmap (str): Color palette.
    """
    if drop_cols:
        data = data.drop(columns=drop_cols, errors="ignore")
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# 5. Regression Analyses and Feature Selection

def regression_and_modeling(data):
    """
    Performs a linear regression analysis on salary, calculates feature importance,
    performs feature selection with SelectKBest (ANOVA and Mutual Info)
    and compares multiple models (Ridge, Lasso, ElasticNet, OLS) using GridSearchCV.
    The data used is from the dummy-encoded DataFrame (data = df_concat).
    """
    print("\n--- Basic Linear Regression ---")
    # Select numeric variables, excluding Salary
    X = data.select_dtypes(include=['int64', 'float64']).drop(columns="Salary", errors="ignore")
    y = data["Salary"]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Linear regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    # Adjusting the coefficients based on the original scale
    original_coefficients = lr_model.coef_ / scaler.scale_
    importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": original_coefficients
    }).sort_values(by="Importance", ascending=False)
    
    print("Feature Importance (adjusted coefficients):")
    print(importances)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Importance", y="Feature", data=importances)
    plt.title("Feature Importance (Linear Regression)")
    plt.xlabel("Adjusted Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
    
    # Feature selection for predicting employee satisfaction (if available)
    if "EmpSatisfaction" in data.columns:
        print("\n--- Feature Selection for EmpSatisfaction ---")
        X_features = data.select_dtypes(include=["number"]).drop(columns=["EmpSatisfaction", "Salary"], errors="ignore")
        Y_target = data["EmpSatisfaction"]
        
        # Impute missing values if any
        imputer = SimpleImputer(strategy="mean")
        X_features = pd.DataFrame(imputer.fit_transform(X_features), columns=X_features.columns)
        
        selector_anova = SelectKBest(score_func=f_classif, k="all")
        selector_anova.fit(X_features, Y_target)
        anova_scores = pd.DataFrame(selector_anova.scores_, index=X_features.columns, columns=["ANOVA Score"])
        
        selector_mutual_info = SelectKBest(score_func=mutual_info_regression, k="all")
        selector_mutual_info.fit(X_features, Y_target)
        mutual_info_scores = pd.DataFrame(selector_mutual_info.scores_, index=X_features.columns, columns=["Mutual Info Score"])
        
        scores = pd.concat([anova_scores, mutual_info_scores], axis=1)
        scores.columns = ["ANOVA Score", "Mutual Info Score"]
        print("Feature importance scores (EmpSatisfaction):")
        print(scores.sort_values(by="ANOVA Score", ascending=False))
    
    # Calculating VIF for numeric variables (on X)
    print("\n--- VIF Calculation ---")
    X_numeric = X.dropna()
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_numeric.columns
    vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) for i in range(X_numeric.shape[1])]
    print(vif_data.sort_values(by="VIF", ascending=False))
    
    # Models with GridSearchCV
    print("\n--- Comparing Regression Models with GridSearchCV ---")
    models = {
        "Ridge": GridSearchCV(Ridge(), param_grid={"alpha": [0.1, 1, 10]}, cv=5),
        "Lasso": GridSearchCV(Lasso(max_iter=10000), param_grid={"alpha": [0.1, 1, 10]}, cv=5),
        "ElasticNet": GridSearchCV(ElasticNet(max_iter=10000), 
                                   param_grid={"alpha": [0.1, 1, 10], "l1_ratio": [0.2, 0.5, 0.8]}, cv=5),
        "OLS": LinearRegression()
    }
    
    results = {}
    for name, mdl in models.items():
        mdl.fit(X_train_scaled, y_train)
        y_pred = mdl.predict(X_test_scaled)
        mse_val = mean_squared_error(y_test, y_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_test, y_pred)
        results[name] = {"MSE": mse_val, "RMSE": rmse_val, "R2": r2_val}
        print(f"{name} - MSE: {mse_val:.2f}, RMSE: {rmse_val:.2f}, R2: {r2_val:.2f}")
    
    # Graphical comparison of results
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].bar(results.keys(), [res["MSE"] for res in results.values()], color=["blue", "green", "red", "purple"])
    ax[0].set_xlabel("Model")
    ax[0].set_ylabel("Mean Squared Error")
    ax[0].set_title("MSE Comparison")
    ax[1].bar(results.keys(), [res["RMSE"] for res in results.values()], color=["blue", "green", "red", "purple"])
    ax[1].set_xlabel("Model")
    ax[1].set_ylabel("Root Mean Squared Error")
    ax[1].set_title("RMSE Comparison")
    ax[2].bar(results.keys(), [res["R2"] for res in results.values()], color=["blue", "green", "red", "purple"])
    ax[2].set_xlabel("Model")
    ax[2].set_ylabel("R-squared")
    ax[2].set_title("R2 Comparison")
    plt.tight_layout()
    plt.show()

# 6. Calling the analysis functions and predictive models

if __name__ == '__main__':
    sns.set(style="whitegrid")
    
    print("\n=== Demographic and Gender Distribution ===")
    plot_demographics(all_data)
    
    print("\n=== Salary and Performance Analysis ===")
    plot_salary_performance(all_data)
    
    print("\n=== Career Progression and Promotions ===")
    plot_career_progression(all_data)
    
    print("\n=== Training and Professional Development ===")
    plot_training_development(all_data)
    
    print("\n=== Employee Turnover and Retention ===")
    analyze_turnover(all_data)
    
    print("\n=== Recruitment Sources and Their Effectiveness ===")
    analyze_recruitment(all_data)
    
    print("\n=== Relationship between Employee Satisfaction and Performance Indicators ===")
    analyze_satisfaction(all_data)
    
    print("\n=== Contract Structure and Its Impact on Performance ===")
    analyze_contracts(all_data)
    
    print("\n=== Analysis of Salary Differences by Gender ===")
    analyze_salary_gender(all_data)
    
    print("\n=== Correlation between Performance and Remuneration by Gender ===")
    analyze_performance_salary_gender(all_data)
    
    print("\n=== Impact of External Factors ===")
    analyze_external_factors(all_data)
    
    print("\n=== Career Progression and Promotions by Gender ===")
    analyze_career_progression_gender(all_data)
    
    print("\n=== Evaluation of Remuneration Policies ===")
    analyze_remuneration_policy(all_data)
    
    print("\n=== Correlation Matrix ===")
    drop_columns = ['Employee_Name', 'State', 'DOB', 'DateofHire', 
                    'DateofTermination', 'TermReason', 'ManagerName', 
                    'RecruitmentSource', 'LastPerformanceReview_Date']
    plot_correlation_matrix(df_concat, title="HR Data Correlation Matrix", drop_cols=drop_columns)
    
    print("\n=== Regression Analyses and Predictive Models ===")
    regression_and_modeling(df_concat)
