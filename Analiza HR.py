import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Importuri pentru regresie și selecția caracteristicilor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Încărcare și Preprocesare Date

url = "https://raw.githubusercontent.com/catamaican/AED/main/HRDataset_v14.csv"

# Citește datasetul
all_data = pd.read_csv(url)

# Afișează câteva informații de bază
print("Primele rânduri din dataset:")
print(all_data.head())
print("\nInformații generale despre dataset:")
all_data.info()
print(f"\nDimensiunea datasetului: {all_data.shape[0]} rânduri și {all_data.shape[1]} coloane.\n")

# Tratarea valorilor nule
all_data['DateofTermination'].fillna('Still Employed', inplace=True)
all_data['ManagerID'].fillna(0, inplace=True)

# Conversia coloanelor de tip dată
all_data['DOB'] = pd.to_datetime(all_data['DOB'], errors='coerce')
all_data['DateofHire'] = pd.to_datetime(all_data['DateofHire'], errors='coerce')
# Pentru DateofTermination, se încearcă conversia; dacă nu e posibil (ex. "Still Employed") se va obține NaT
all_data['DateofTermination_converted'] = pd.to_datetime(all_data['DateofTermination'], errors='coerce')

# Calculul vârstei și a vechimii în companie
today = pd.Timestamp('today')
all_data['Age'] = ((today - all_data['DOB']).dt.days / 365).astype(int)
all_data['Experience'] = ((today - all_data['DateofHire']).dt.days / 365).astype(int)

# Mapare pentru scorul de performanță
performance_map = {'Exceeds': 3, 'Meets': 2, 'Needs Improvement': 1, 'PIP': 0}
all_data['PerformanceNumeric'] = all_data['PerformanceScore'].map(performance_map)

# Indicator turnover: 0 = încă angajat, 1 = a părăsit compania
all_data['Turnover'] = np.where(all_data['DateofTermination'] == 'Still Employed', 0, 1)

# Pentru unele analize, se dorește ca genul să rămână textual; 
# dacă este nevoie de reprezentare numerică, se poate crea o coloană suplimentară.
all_data['Sex_numeric'] = all_data['Sex'].apply(lambda x: 1 if x == 'M' else 0)


# 2. Transformări suplimentare: Dummy Encoding

# Realizăm dummy encoding pentru variabilele categorice selectate
dummy_cols = ["MaritalDesc", "Department", "Position", "CitizenDesc", "RaceDesc", "EmploymentStatus", "PerformanceScore"]
df_dummies = pd.get_dummies(all_data, columns=dummy_cols, drop_first=True)

# Mapare pentru coloanele binare din dummy (ex. HispanicLatino și Sex)
df_dummies['HispanicLatino'] = df_dummies['HispanicLatino'].apply(lambda x: 1 if x == 'Yes' else 0)
df_dummies['Sex'] = df_dummies['Sex'].apply(lambda x: 1 if x == 'M' else 0)

# Concatenăm datele originale cu cele dummy-encoded
df_concat = pd.concat([all_data, df_dummies], axis=1)
# Eliminăm duplicatele de coloană
df_concat = df_concat.loc[:, ~df_concat.columns.duplicated()]
# Dacă se dorește, eliminăm coloanele originale care au fost dummy-encode
df_concat = df_concat.drop(columns=dummy_cols, errors="ignore")

# 3. Statistici Descriptive și Vizualizări Inițiale

# Afisăm statistici descriptive pentru df_concat
print("\nStatistici descriptive (df_concat):")
print(df_concat.info())
print(df_concat.describe())

numeric_columns_all_data = all_data.select_dtypes(include=[np.number])
median_values = numeric_columns_all_data.median()
print(f"Salariu - Mediana: {median_values['Salary']:.2f}")
print(f"Satisfacția angajaților - Mediana: {median_values['EmpSatisfaction']:.2f}")

# Selectăm coloanele numerice și calculăm cvartilele
numeric_columns = df_concat.select_dtypes(include=[np.number])
quartiles = numeric_columns.quantile([0.25, 0.5, 0.75])
print("\nCvartilele:\n", quartiles)
if 'Salary' in numeric_columns.columns:
    print(f"Salariu - Cvartila 25%: {quartiles.loc[0.25, 'Salary']:.2f}")
    print(f"Salariu - Cvartila 50%: {quartiles.loc[0.5, 'Salary']:.2f}")
    print(f"Salariu - Cvartila 75%: {quartiles.loc[0.75, 'Salary']:.2f}")
else:
    print("Coloana 'Salary' nu există în dataset.")
if 'EmpSatisfaction' in numeric_columns.columns:
    print(f"Satisfacție angajat - Cvartila 25%: {quartiles.loc[0.25, 'EmpSatisfaction']:.2f}")
    print(f"Satisfacție angajat - Cvartila 50%: {quartiles.loc[0.5, 'EmpSatisfaction']:.2f}")
    print(f"Satisfacție angajat - Cvartila 75%: {quartiles.loc[0.75, 'EmpSatisfaction']:.2f}")
else:
    print("Coloana 'EmpSatisfaction' nu există în dataset.")

# Vizualizarea distribuției salariilor
sns.histplot(all_data['Salary'], kde=True)
plt.title("Distribuția Salariilor")
plt.show()

sns.boxplot(x=all_data["Department"], y=all_data["Salary"])
plt.title("Distribuția salariilor pe departamente")
plt.xlabel("Departament")
plt.ylabel("Salariu")
plt.xticks(rotation=45)
plt.show()
# Graficul histograma cu KDE arată modul în care salariile sunt distribuite – de regulă, distribuția poate fi asimetrică, indicând prezența unor extreme (salarii foarte mari comparativ cu majoritatea).

# Relația dintre absențe și salariu
plt.scatter(all_data["Absences"], all_data["Salary"], alpha=0.5)
plt.title("Relația dintre absențe și salariu")
plt.xlabel("Numărul de absențe")
plt.ylabel("Salariu")
plt.show()

# Calculăm corelația Pearson între absențe și salariu
corr, p_value = pearsonr(all_data["Absences"].dropna(), all_data["Salary"].dropna())
print(f"Coeficientul de corelație Pearson: {corr:.2f}")
print(f"P-value: {p_value:.4f}")
#Graficul de tip scatter sugerează o relație între numărul de absențe și salariu.
#Calculul coeficientului Pearson (valoare și p-value) oferă o măsură cantitativă: un coeficient mai aproape de 0 indică o relație slabă, în timp ce o valoare semnificativ diferită de 0 (și un p-value mic) ar sugera o legătură mai puternică.

# Sursa principală de recrutare
recruitment_source_counts = all_data["RecruitmentSource"].value_counts()
print("Numărul de angajați recrutați pe surse:\n", recruitment_source_counts)
sns.barplot(x=recruitment_source_counts.index, y=recruitment_source_counts.values, ci=None)
plt.title("Distribuția surselor de recrutare")
plt.xlabel("Sursa de recrutare")
plt.ylabel("Număr de angajați")
plt.xticks(rotation=45)
plt.show()

# Top 5 angajați după salariu
if {"Employee_Name", "Position", "Salary"}.issubset(all_data.columns):
    top_employees = all_data[["Employee_Name", "Position", "Salary"]].sort_values(by="Salary", ascending=False).head(5)
    print("Top 5 angajați cu cele mai mari salarii:")
    print(top_employees)

# 4. Funcții de Analiză și Vizualizare

def plot_demographics(data):
    """
    Distribuția demografică: vârstă și gen în cadrul fiecărui departament și poziție.
    """
    plt.figure(figsize=(12,6))
    sns.boxplot(data=data, x='Department', y='Age', hue='Sex')
    plt.title("Distribuția vârstei pe departamente (colorare după gen)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,6))
    sns.boxplot(data=data, x='Position', y='Age', hue='Sex')
    plt.title("Distribuția vârstei pe poziții (colorare după gen)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_salary_performance(data):
    """
    Analiza salarială: salariul mediu pe departamente, relația salariu vs. experiență și
    salariu vs. scorul de performanță; plus analiză pe categorii de experiență.
    """
    dept_salary = data.groupby("Department")["Salary"].mean().reset_index()
    plt.figure(figsize=(10,6))
    sns.barplot(data=dept_salary, x="Department", y="Salary")
    plt.title("Salariu mediu pe departamente")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="Experience", y="Salary", hue="Department")
    plt.title("Corelația între experiență și salariu")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="PerformanceNumeric", y="Salary", hue="Department")
    plt.title("Corelația între scorul de performanță și salariu")
    plt.tight_layout()
    plt.show()
    
    data['ExperienceCat'] = pd.cut(data['Experience'], bins=[0,5,10,20,50], labels=["0-5","5-10","10-20","20+"])
    plt.figure(figsize=(8,6))
    sns.boxplot(data=data, x="ExperienceCat", y="Salary")
    plt.title("Distribuția salariilor pe categorii de experiență")
    plt.tight_layout()
    plt.show()

def plot_career_progression(data):
    """
    Evoluția carierei: relația dintre vechime și salariu, diferențiată după poziție și indicatorul turnover.
    """
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="Experience", y="Salary", hue="Position", style="Turnover")
    plt.title("Salariu vs. Vechime (indicativ pentru evoluția carierei)")
    plt.tight_layout()
    plt.show()

def plot_training_development(data):
    """
    Analiza trainingului și dezvoltării: distribuția numărului de proiecte speciale (proxy pentru training)
    și relația cu satisfacția angajaților.
    """
    if "SpecialProjectsCount" in data.columns:
        plt.figure(figsize=(8,6))
        sns.histplot(data["SpecialProjectsCount"], kde=True, bins=20)
        plt.title("Distribuția numărului de proiecte speciale")
        plt.tight_layout()
        plt.show()
    
        plt.figure(figsize=(8,6))
        sns.boxplot(data=data, x="SpecialProjectsCount", y="EmpSatisfaction")
        plt.title("Satisfacția angajaților vs. numărul de proiecte speciale")
        plt.tight_layout()
        plt.show()
    else:
        print("Coloana 'SpecialProjectsCount' nu este disponibilă în dataset.")

def analyze_turnover(data):
    """
    Analiza turnover-ului: rata medie de turnover pe departamente.
    """
    dept_turnover = data.groupby("Department")["Turnover"].mean().reset_index()
    plt.figure(figsize=(10,6))
    sns.barplot(data=dept_turnover, x="Department", y="Turnover")
    plt.title("Rata de turnover pe departamente")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_recruitment(data):
    """
    Analiza surselor de recrutare: performanța (numeric) și rata turnover-ului în funcție de sursa de recrutare.
    """
    if "RecruitmentSource" in data.columns:
        plt.figure(figsize=(12,6))
        sns.boxplot(data=data, x="RecruitmentSource", y="PerformanceNumeric")
        plt.title("Performanță după sursa de recrutare")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        rec_turnover = data.groupby("RecruitmentSource")["Turnover"].mean().reset_index()
        plt.figure(figsize=(12,6))
        sns.barplot(data=rec_turnover, x="RecruitmentSource", y="Turnover")
        plt.title("Rata de turnover după sursa de recrutare")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Coloana 'RecruitmentSource' nu este disponibilă.")

def analyze_satisfaction(data):
    """
    Relația dintre satisfacția angajaților și indicatorii de performanță și absențe.
    Se calculează și coeficienții de corelație Pearson.
    """
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="PerformanceNumeric", y="EmpSatisfaction", hue="Department")
    plt.title("Satisfacția angajaților vs. scorul de performanță")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="Absences", y="EmpSatisfaction", hue="Department")
    plt.title("Satisfacția angajaților vs. absențe")
    plt.tight_layout()
    plt.show()
    
    if data["PerformanceNumeric"].notnull().sum() > 0 and data["EmpSatisfaction"].notnull().sum() > 0:
        corr_perf, _ = pearsonr(data["PerformanceNumeric"].dropna(), data["EmpSatisfaction"].dropna())
        print(f"Corelația între satisfacție și performanță: {corr_perf:.2f}")
    if data["Absences"].notnull().sum() > 0 and data["EmpSatisfaction"].notnull().sum() > 0:
        corr_abs, _ = pearsonr(data["Absences"].dropna(), data["EmpSatisfaction"].dropna())
        print(f"Corelația între satisfacție și absențe: {corr_abs:.2f}")

def analyze_contracts(data):
    """
    Analiza structurii contractuale: distribuția tipurilor de contract și impactul asupra performanței și turnover-ului.
    """
    if "EmploymentStatus" in data.columns:
        status_counts = data["EmploymentStatus"].value_counts().reset_index()
        status_counts.columns = ["EmploymentStatus", "Count"]
        plt.figure(figsize=(8,6))
        sns.barplot(data=status_counts, x="EmploymentStatus", y="Count")
        plt.title("Distribuția tipurilor de contract")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(12,6))
        sns.boxplot(data=data, x="EmploymentStatus", y="PerformanceNumeric")
        plt.title("Performanța după tipul de contract")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        status_turnover = data.groupby("EmploymentStatus")["Turnover"].mean().reset_index()
        plt.figure(figsize=(8,6))
        sns.barplot(data=status_turnover, x="EmploymentStatus", y="Turnover")
        plt.title("Rata de turnover după tipul de contract")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Coloana 'EmploymentStatus' nu este disponibilă.")

def analyze_salary_gender(data):
    """
    Analiza diferențelor salariale pe gen: distribuția salariilor și salariul mediu pe gen.
    """
    plt.figure(figsize=(8,6))
    sns.boxplot(data=data, x="Sex", y="Salary")
    plt.title("Distribuția salariilor pe gen")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12,6))
    sns.boxplot(data=data, x="Department", y="Salary", hue="Sex")
    plt.title("Salariu pe departament, diferențiat după gen")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    mean_salary = data.groupby("Sex")["Salary"].mean()
    print("Salariu mediu pe gen:")
    print(mean_salary)

def analyze_performance_salary_gender(data):
    """
    Corelația dintre performanță și salariu, analizată separat pe gen.
    """
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="PerformanceNumeric", y="Salary", hue="Sex")
    plt.title("Corelația între performanță și salariu (după gen)")
    plt.tight_layout()
    plt.show()
    
    for gender in data['Sex'].unique():
        subset = data[data['Sex'] == gender]
        if subset["PerformanceNumeric"].notnull().sum() > 0 and subset["Salary"].notnull().sum() > 0:
            corr, _ = pearsonr(subset["PerformanceNumeric"].dropna(), subset["Salary"].dropna())
            print(f"Corelația între performanță și salariu pentru gen {gender}: {corr:.2f}")

def analyze_external_factors(data):
    """
    Impactul factorilor externi: influența experienței (și, dacă există, a proiectelor speciale)
    asupra salariului, diferențiat pe gen.
    """
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="Experience", y="Salary", hue="Sex")
    plt.title("Impactul experienței asupra salariului (după gen)")
    plt.tight_layout()
    plt.show()
    
    if "SpecialProjectsCount" in data.columns:
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=data, x="SpecialProjectsCount", y="Salary", hue="Sex")
        plt.title("Impactul proiectelor speciale asupra salariului (după gen)")
        plt.tight_layout()
        plt.show()

def analyze_career_progression_gender(data):
    """
    Evoluția carierei diferențiată pe gen: relația dintre vechime și salariu pentru bărbați și femei.
    """
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="Experience", y="Salary", hue="Sex")
    plt.title("Evoluția carierei: Salariu vs. Vechime (după gen)")
    plt.tight_layout()
    plt.show()

def analyze_remuneration_policy(data):
    """
    Evaluarea politicilor de remunerare: cum se distribuie salariile în funcție de scorul de performanță și gen,
    precum și relația salariu vs. satisfacția angajaților.
    """
    plt.figure(figsize=(12,6))
    sns.boxplot(data=data, x="PerformanceScore", y="Salary", hue="Sex")
    plt.title("Distribuția salariilor pe scorul de performanță și gen")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x="EmpSatisfaction", y="Salary", hue="Sex")
    plt.title("Salariu vs. Satisfacția angajaților (după gen)")
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(data, title="Matricea de corelație", drop_cols=None, figsize=(12, 10), cmap="coolwarm"):
    """
    Calculează și afișează matricea de corelație pentru variabilele numerice dintr-un DataFrame.
    Parametri:
      data (pd.DataFrame): DataFrame-ul pentru care se calculează corelațiile.
      title (str): Titlul graficului.
      drop_cols (list, optional): Listă de coloane de eliminat înainte de calcul.
      figsize (tuple): Dimensiunea figurii.
      cmap (str): Paleta de culori.
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

# 5. Analize de Regresie și Selecție a Caracteristicilor

def regression_and_modeling(data):
    """
    Efectuează o analiză de regresie liniară pe salariu, calculează importanța caracteristicilor,
    efectuează selecția caracteristicilor cu SelectKBest (ANOVA și Mutual Info)
    și compară mai multe modele (Ridge, Lasso, ElasticNet, OLS) folosind GridSearchCV.
    Datele folosite sunt din DataFrame-ul cu dummy encoding (data = df_concat).
    """
    print("\n--- Regresie Liniară de bază ---")
    # Selectăm variabilele numerice, excluzând Salary
    X = data.select_dtypes(include=['int64', 'float64']).drop(columns="Salary", errors="ignore")
    y = data["Salary"]
    
    # Împărțirea datelor
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardizare
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelul de regresie liniară
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    # Ajustarea coeficienților în funcție de scara originală
    original_coefficients = lr_model.coef_ / scaler.scale_
    importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": original_coefficients
    }).sort_values(by="Importance", ascending=False)
    
    print("Importanța caracteristicilor (coeficienți ajustați):")
    print(importances)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Importance", y="Feature", data=importances)
    plt.title("Importanța caracteristicilor (Regresie liniară)")
    plt.xlabel("Coeficient ajustat")
    plt.ylabel("Caracteristică")
    plt.tight_layout()
    plt.show()
    
    # Selecția caracteristicilor pentru prezicerea satisfacției angajaților (dacă există)
    if "EmpSatisfaction" in data.columns:
        print("\n--- Selecție caracteristici pentru EmpSatisfaction ---")
        X_features = data.select_dtypes(include=["number"]).drop(columns=["EmpSatisfaction", "Salary"], errors="ignore")
        Y_target = data["EmpSatisfaction"]
        
        # Imputăm eventualele valori lipsă
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
        print("Scorurile pentru importanța caracteristicilor (EmpSatisfaction):")
        print(scores.sort_values(by="ANOVA Score", ascending=False))
    
    # Calculul VIF pentru variabilele numerice (pe X)
    print("\n--- Calcul VIF ---")
    X_numeric = X.dropna()
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_numeric.columns
    vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) for i in range(X_numeric.shape[1])]
    print(vif_data.sort_values(by="VIF", ascending=False))
    
    # Modele cu GridSearchCV
    print("\n--- Compararea modelelor de regresie cu GridSearchCV ---")
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
    
    # Compararea grafică a rezultatelor
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

# 6. Apelul funcțiilor de analiză și modele predictive

if __name__ == '__main__':
    sns.set(style="whitegrid")
    
    print("\n=== Distribuția Demografică și de Gen ===")
    plot_demographics(all_data)
    
    print("\n=== Analiza Salarială și a Performanței ===")
    plot_salary_performance(all_data)
    
    print("\n=== Evoluția Carierelor și Promovărilor ===")
    plot_career_progression(all_data)
    
    print("\n=== Training și Dezvoltare Profesională ===")
    plot_training_development(all_data)
    
    print("\n=== Fluctuația și Retenția Angajaților ===")
    analyze_turnover(all_data)
    
    print("\n=== Surse de Recrutare și Eficacitatea Lor ===")
    analyze_recruitment(all_data)
    
    print("\n=== Relația Între Satisfacția Angajaților și Indicatorii de Performanță ===")
    analyze_satisfaction(all_data)
    
    print("\n=== Structura Contractuală și Impactul Asupra Performanței ===")
    analyze_contracts(all_data)
    
    print("\n=== Analiza Diferențelor Salariale pe Gen ===")
    analyze_salary_gender(all_data)
    
    print("\n=== Corelația Între Performanță și Remunerație pe Gen ===")
    analyze_performance_salary_gender(all_data)
    
    print("\n=== Impactul Factorilor Externi ===")
    analyze_external_factors(all_data)
    
    print("\n=== Evoluția Carierelor și Promovărilor după Gen ===")
    analyze_career_progression_gender(all_data)
    
    print("\n=== Evaluarea Politicilor de Remunerație ===")
    analyze_remuneration_policy(all_data)
    
    print("\n=== Matricea de Corelație ===")
    drop_columns = ['Employee_Name', 'State', 'DOB', 'DateofHire', 
                    'DateofTermination', 'TermReason', 'ManagerName', 
                    'RecruitmentSource', 'LastPerformanceReview_Date']
    plot_correlation_matrix(df_concat, title="Matricea de corelație a datelor HR", drop_cols=drop_columns)
    
    print("\n=== Analize de Regresie și Modele Predictive ===")
    regression_and_modeling(df_concat)
