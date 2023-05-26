import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
df= pd.read_csv('dataset.csv',sep=',')
df= df.dropna()
df=df.dropna(axis=1)
df=df.drop('Debtor',axis=1)

# Categorize the features into the different categories
demographic_data = ['Marital status', 'Nacionality', 'Displaced', 'Gender', 'Age at enrollment', 'International']
socioeconomic_data = ["Father's qualification", "Mother's qualification", "Father's occupation", "Mother's occupation", "Educational special needs", "Debtor", "Tuition fees up to date", "Scholarship holder"]
macroeconomic_data = ['Unemployment rate', 'GDP', 'Inflation rate']
academic_data_enrollment = ['Application mode', 'Application order', 'Course', 'Daytime/evening attendance', 'Previous qualification']
academic_data_1st_sem = ['Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)']
academic_data_2nd_sem = ['Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)']
target_variable = 'Target'

# Load the data and split it into train and test sets
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a Random Forest model and calculate feature importances
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_importances = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
# Sort the feature importances and extract the top 3 features for each category
demographic_top3 = [demographic_data[i] for i in rf_importances.importances_mean[:len(demographic_data)].argsort()[::-1][:3]]
socioeconomic_top4 = [socioeconomic_data[i] for i in rf_importances.importances_mean[len(demographic_data):len(demographic_data)+len(socioeconomic_data)].argsort()[::-1][:4]]
macroeconomic_top1 = [macroeconomic_data[i] for i in rf_importances.importances_mean[len(demographic_data)+len(socioeconomic_data):len(demographic_data)+len(socioeconomic_data)+len(macroeconomic_data)].argsort()[::-1][:1]]
enrollment_top2 = [academic_data_enrollment[i] for i in rf_importances.importances_mean[len(demographic_data)+len(socioeconomic_data)+len(macroeconomic_data):len(demographic_data)+len(socioeconomic_data)+len(macroeconomic_data)+len(academic_data_enrollment)].argsort()[::-1][:2]]
sem1_top4 = [academic_data_1st_sem[i] for i in rf_importances.importances_mean[len(demographic_data)+len(socioeconomic_data)+len(macroeconomic_data)+len(academic_data_enrollment):len(demographic_data)+len(socioeconomic_data)+len(macroeconomic_data)+len(academic_data_enrollment)+len(academic_data_1st_sem)].argsort()[::-1][:4]]
sem2_top4 = [academic_data_2nd_sem[i] for i in rf_importances.importances_mean[len(demographic_data)+len(socioeconomic_data)+len(macroeconomic_data)+len(academic_data_enrollment):len(demographic_data)+len(socioeconomic_data)+len(macroeconomic_data)+len(academic_data_enrollment)+len(academic_data_2nd_sem)].argsort()[::-1][:4]]

keep_features = demographic_top3 +  socioeconomic_top4 + macroeconomic_top1 + enrollment_top2 + sem1_top4 +sem2_top4

X=df[keep_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#scale the data
#scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train the decision tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# make predictions on the test set
y_pred = dt.predict(X_test)

# save the model
pickle.dump(dt,open('model.pkl','wb'))
