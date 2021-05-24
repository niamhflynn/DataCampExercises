#######################
# IMPORTS
#######################
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, \
    f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import plot_partial_dependence
SEED = 13

#######################
# READ DATA
#######################
# from kaggle.api.kaggle_api_extended import KaggleApi
# api = KaggleApi()
# api.authenticate()

# api.dataset_download_file('blastchar/telco-customer-churn',
#                          file_name='WA_Fn-UseC_-Telco-Customer-Churn.csv')


def import_data(url):
    """Creates dataframe from url of csv for data .

    Args:
       url (string): Url of csv data to import from current workspace.

     Returns:
       Imported dataset as a pandas df.
     """
    data = pd.read_csv(url)
    return data


churn_data = import_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")


def get_dataset_info(df):
    """Creates dataframe from url of csv for data .

     Args:
       df (Pandas dataframe): DataFrame we want to analyse.

     Returns:
       Dictionary of dataset information, including:.
        Head, Describe, Shape and Info
     """
    info = {
        'head': df.head(),
        'describe': df.describe(),
        'shape': df.shape,
        'info': df.info()
    }
    return info


churn_info = get_dataset_info(churn_data)
print(churn_data.columns)
print(churn_info['info'])

# From above info, we can see missing information in Total Charges col - we will remove this.
churn_data = churn_data[churn_data.TotalCharges != ' ']
churn_data.TotalCharges = pd.to_numeric(churn_data.TotalCharges)


def print_unique_col_values(df):
    """Prints all unique values stored in each dataset column.

     Args:
     df (pandas DataFrame): The data to examine.

      Returns:
      Printed list of values for each column in the dataset.
        """
    for column in df:
        if df[column].dtypes == 'object':
            print(f'{column}: {df[column].unique()}')


print_unique_col_values(churn_data)
#######################
# CLEAN DATA
#######################
# check for duplicated values
churn_data.duplicated(keep='last')
print(churn_data.shape)

# check for NA values
print(churn_data.isna().sum())

# Encode object variables
col_name = ['SeniorCitizen', 'Churn']
churn_data[col_name] = churn_data[col_name].astype(object)

#######################
# EXPLORATORY ANALYSIS
#######################
# set colourscheme for plots sns.set_theme(style="whitegrid")
sns.color_palette("magma", as_cmap=True)


# PLOT ONE - WHAT PROPORTION OF CUSTOMERS CHURN


def demographics_vs_churn(df, target_col, demographic_col):
    """Print graph for demographic variable vs the target variable

     Args:
     churn_data (pandas DataFrame): The data being examined containing target_col and demographic_col
     target_col (string): The target variable for the analysis
     demographic_col (string): The demographic variable of interest

      Returns:
      Plot of proportion of customers who churn for each demographic
        """
    return sns.countplot(x=demographic_col, hue=target_col, data=df)


# gender_vs_churn = demographics_vs_churn(churn_data, "Churn", "gender")
# partner_vs_churn = demographics_vs_churn(churn_data, "Churn", "Partner")
# dependents_vs_churn = demographics_vs_churn(churn_data, "Churn", "Dependents")
churn_data['SeniorCitizen'] = churn_data['SeniorCitizen'].map(
    {0: 'No', 1: 'Yes'})


# srcitizen_vs_churn = demographics_vs_churn(churn_data, "Churn", "SeniorCitizen")
# monthly_churn = sns.countplot(x="tenure", data=churn_data, hue="Churn")
# Demographics

# PLOT AVERAGE MONTHLY CHARGES VS TENURE FOR 2 CATEGORIES - CHURN AND NOT CHURN
# Draw a nested barplot by species and sex
# monthly_charges_vs_tenure = sns.relplot(x="tenure", y="MonthlyCharges", hue="Churn",
#                                        kind="line", data=churn_data)


# plt.show()
# BOXPLOT CHURN AND TENURE
# tenure_vs_churn = sns.boxplot(y="tenure", x="Churn", palette=["m", "g"], data=churn_data)
# sns.despine(offset=10, trim=True)
# plt.show()


def kdeplot(feature):
    """
    This function creates a Kernel Density Plot for the inputted feature using the churn dataframe. Two plots are
    created for each value of Churn (i.e. yes and no)
    :param feature: feature for which we want to create a KDE plot with the churn dataframe
    :return: Two KDE plots for inputted feature and each value of Churn (i.e. yes or no)
    """
    plt.figure(figsize=(9, 4))
    plt.title("KDE for {}".format(feature))
    ax0 = sns.kdeplot(churn_data[churn_data['Churn'] == 'No'][feature].dropna(), color='navy', label='Churn: No')
    ax1 = sns.kdeplot(churn_data[churn_data['Churn'] == 'Yes'][feature].dropna(), color='orange', label='Churn: Yes')


# kdeplot('tenure')
# kdeplot('MonthlyCharges')
# churn_data['TotalCharges'] = churn_data['TotalCharges'].replace(" ", 0).astype('float32')
# kdeplot('TotalCharges')
# plt.show()

#######################
# CLEANING FOR ANALYSIS
#######################
# Clean binary variables
# After examining output from "print_unique_col_values" function, it is shown that entries "No Phone Service" or
# "No internet service" can be replaced with "No"
churn_data.replace('No internet service', 'No', inplace=True)
churn_data.replace('No phone service', 'No', inplace=True)
binary_columns = ['Partner', 'SeniorCitizen', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'PaperlessBilling', 'Churn']


def replace_binary_columns(df, binary_cols):
    """
    This function creates a Kernel Density Plot for the inputted feature using the churn dataframe. Two plots are
    created for each value of Churn (i.e. yes and no)
    :df (Pandas DataFrame): dataframe that the binary values are in
    :binary_cols (list of column names): Columns from the df DataFrame that contain the values "Yes" and "No", that we
     want to update to 1 and 0 respectively
    """
    for col in binary_cols:
        df[col].replace({'Yes': 1, 'No': 0}, inplace=True)


replace_binary_columns(churn_data, binary_columns)
churn_data_original = churn_data.copy()

# ENCODING
# change value for male to 1 and female to 0 for gender variable
gender_encoder = LabelEncoder()
churn_data['gender_label'] = gender_encoder.fit_transform(churn_data['gender'])
churn_features = churn_data.drop(['customerID', 'gender'], axis='columns')
churn_features = pd.get_dummies(data=churn_features, columns=['InternetService', 'Contract', 'PaymentMethod'])
print(churn_features[churn_features.columns[1:]].corr()['Churn'][:].sort_values(ascending=False))
# DROP TOTAL CHARGES AS IT IS A PRODUCT OF MONTHLY CHARGES AND TENURE
churn_features = churn_features.drop(['Churn', 'TotalCharges'], axis='columns')

#######################
# NORMALISE
#######################
def normalise_smote_test_split(df, test_size):
    """
    This function scales the data using MinMaxScaler(), SMOTE is used to create balanced data and the data is separated
     into training and test splits - this function is tailored to Churn dataset with target "Churn"
    :df (Pandas DataFrame): dataframe to be scaled and split
    :test_size (decimal): Proportion of values to be allocated to test data
    """
    features_scaler = MinMaxScaler()
    features = features_scaler.fit_transform(df)
    x = pd.DataFrame(features, columns=churn_features.columns)
    feature_labels = x.columns.values
    print("THESE ARE THE CHURNFEATURES.columns as columns")
    print(feature_labels)
    y = churn_data_original.Churn
    #######################
    # SMOTE
    #######################
    smote = SMOTE(sampling_strategy='minority')
    x_smote, y_smote = smote.fit_resample(x, y)

    #######################
    # MACHINE LEARNING
    #######################
    # split the data to training and test data
    # Set SEED for reproducibility
    SEED = 13
    X_train, X_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size=test_size, random_state=SEED)
    return X_train, X_test, y_train, y_test


######################
# Feature Importance
######################
# Initial Model created to determine important features (remove unimportant features to reduce noise)
X_train, X_test, y_train, y_test = normalise_smote_test_split(churn_features, 0.3)

def create_feature_importance_plot(model, df):
    """
    This function creates a feature importance plot from a predictive model and returns feature importance pandas
    series
    :model: The predictive model used for the analysis
    :df: The original dataframe that the model was created
    from - will be used to get column names for each variable
    """
    # plot feature importance
    important_features = pd.Series(data=model.feature_importances_, index=df.columns.values)
    important_features = important_features.sort_values(ascending=False)
    important_features.plot.barh(x='data', y=important_features.index)
    plt.title('Feature Importance')
    plt.ylabel('Feature')
    plt.xlabel('Importance')
    plt.show()
    return important_features

SEED=13
feature_importance_rf = RandomForestClassifier(random_state=SEED)
feature_importance_rf.fit(X_train, y_train)
create_feature_importance_plot(feature_importance_rf, churn_features)
all_feat_pred = feature_importance_rf.predict(X_test)
print('Accuracy Score All features: ' + str(accuracy_score(y_test, all_feat_pred)))
# Keep only the columns with over 80 importance
fi = pd.DataFrame({'cols': churn_features.columns.values, 'feature-importances':
    feature_importance_rf.feature_importances_})
print("Printing fi")
df_keep = pd.DataFrame(fi['feature-importances'] > 0.025)

for i in range(len(df_keep)):
    if not df_keep['feature-importances'][i]:
        churn_features = churn_features.drop(fi['cols'][i], axis='columns')

#X_train, X_test, y_train, y_test = normalise_smote_test_split(churn_features, 0.3)
#removed_features_rf = RandomForestClassifier(random_state=SEED)
#removed_features_rf.fit(X_train, y_train)
#removed_features_pred = removed_features_rf.predict(X_test)
#print('Accuracy Score removed features: ' + str(accuracy_score(y_test, removed_features_pred)))
#print('Improvement: ' + str(accuracy_score(y_test, removed_features_pred) - accuracy_score(y_test, all_feat_pred)))

# FOUND THAT REMOVING VARIABLES DID NOT LEAD TO IMPROVEMENT, WILL KEEP ALL VARIABLES FOR ANALYSIS.
#######################
# LOGISTIC REGRESSION
########################


def churn_logistic_regression(X_train, X_test, y_train, y_test):
    """
    This function creates a logistic regression model for the inputted data and uses hyperparameter tuning to
    find the model paramaters that have give the greatest accuracy
    :X_train: Training set of independent features for machine learning
    :X_test: Test set of independent features for machine learning
    :Y_train: Training set of dependent / target variables
    :Y_test: Test set of dependent / target variables
    """
    logreg = LogisticRegression(solver='liblinear')
    grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}  # l1 lasso l2 ridge
    logreg_cv = GridSearchCV(logreg, grid, cv=10)
    logreg_cv.fit(X_train, y_train)

    # CREATE LOGISTIC REGRESSION MODEL BASED OFF BEST RESULTS ABOVE
    logreg = LogisticRegression(solver='liblinear', C=logreg_cv.best_params_['C'],
                                penalty=logreg_cv.best_params_['penalty'], max_iter=10000)

    # Fit the classifier to the training data
    logreg.fit(X_train, y_train)
    # Predict the labels of the test set: y_pred
    y_pred = logreg.predict(X_test)
    # Compute and print the confusion matrix and classification report

    print(classification_report(y_test, y_pred))
    # AUC
    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = logreg.predict_proba(X_test)[:, 1]

    # Compute and print AUC score
    print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

    # Compute cross-validated AUC scores: cv_auc
    cv_auc = cross_val_score(logreg, x_smote, y_smote, cv=5, scoring='roc_auc')

    y_pred = logreg.predict(X_test)
    # Print list of AUC scores
    print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))
    # New Model Evaluation metrics
    print('Accuracy Score : ' + str(accuracy_score(y_test, y_pred)))
    print('Precision Score : ' + str(precision_score(y_test, y_pred)))
    print('Recall Score : ' + str(recall_score(y_test, y_pred)))
    print('F1 Score : ' + str(f1_score(y_test, y_pred)))
    logistic_regression_info = {
        'best_params': logreg_cv.best_params_,
        'cv_accuracy': logreg_cv.best_score_,
        'model': logreg,
        'accuracy': str(accuracy_score(y_test, y_pred)),
        'precision': str(precision_score(y_test, y_pred)),
        'recall': str(recall_score(y_test, y_pred)),
        'f1': str(f1_score(y_test, y_pred)),
        'cv_auc': cv_auc,
        'predicted_values': y_pred
    }
    return logistic_regression_info


def create_confusion_matrix(actual, predicted):
    """
        This function creates a confusion matrix based off of inputted actual and predicted values of the
        target variable
        :actual: Training set of independent features for machine learning
        :predicted: Test set of independent features for machine learning
        """
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt=".1f")
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()


def create_roc_curve(actual, predicted):
    """
        This function creates a confusion matrix based off of inputted actual and predicted values of the
        target variable
        :actual: Actual values of the target variable in test set
        :predicted: Predicted values of the target variables using the test feature variables
        """
    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(actual, predicted)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()


# logistic_regression_model = churn_logistic_regression(X_train, X_test, y_train, y_test)
# model = logistic_regression_model['model']

#######################
# CLASSIFICATION TREE
########################
dt = DecisionTreeClassifier(random_state=SEED)
grid = {"criterion": ['gini', 'entropy'], "max_depth": [8, 9, 10, 11, 12, 13]}

# dt_cv = GridSearchCV(dt, grid, cv=5)
# dt_cv.fit(X_train, y_train)
# dt_cv.fit(X_train, y_train)

# print("tuned hpyerparameters :(best parameters) ", dt_cv.best_params_)
# print("accuracy :", dt_cv.best_score_)
# RESULTS ARE ENTROPY AND 10
dt_final = DecisionTreeClassifier(random_state=13,
                                  max_depth=10,
                                  criterion='gini')

dt_final.fit(X_train, y_train)
dt_pred = dt_final.predict(X_test)
print('Accuracy Score : ' + str(accuracy_score(y_test, dt_pred)))
print('Precision Score : ' + str(precision_score(y_test, dt_pred)))
print('Recall Score : ' + str(recall_score(y_test, dt_pred)))
print('F1 Score : ' + str(f1_score(y_test, dt_pred)))

# fig = plt.figure(figsize=(25, 20))
# _ = tree.plot_tree(dt_final,
#                   filled=True)
# plt.savefig('foo.png')

#######################
# RANDOM FOREST
########################
rf = RandomForestClassifier(random_state=SEED)
param_grid = {
    'bootstrap': [True],
    'max_depth': [1, 10, 20, 30],
    'max_features': [2, 3, 6],
    'min_samples_leaf': [1, 3, 4, 5],
    'min_samples_split': [2, 4, 6],
    'n_estimators': [100, 300, 500]
}
# grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid,
# cv=5, n_jobs=-1, verbose=2)
# grid_search_rf.fit(X_train, y_train)
# print("tuned hpyerparameters :(best parameters) ", grid_search_rf.best_params_)
# print("tuned hpyerparameters :(best parameters) ", grid_search_rf.best_params_)
# BEST PARAMS = BOOTSTRAP TRUE, MAX_DEPTH 80, MAX_FEATURES 3, MIN_SAMPLES_LEAF 3, MIN_SAMPLES_SPLIT 8,
# N_ESTIMATORS = 1000
rf_tuned = RandomForestClassifier(random_state=13, bootstrap=True, max_depth=20, max_features=2, min_samples_leaf=1,
                                  min_samples_split=4, n_estimators=300)
rf_tuned.fit(X_train, y_train)
rf_pred = rf_tuned.predict(X_test)
print('RF Accuracy Score : ' + str(accuracy_score(y_test, rf_pred)))
print('RF Precision Score : ' + str(precision_score(y_test, rf_pred)))
print('RF Recall Score : ' + str(recall_score(y_test, rf_pred)))
print('RF F1 Score : ' + str(f1_score(y_test, rf_pred)))
# plot feature importance for random forest model
# rf_feature_importance = create_feature_importance_plot(rf_tuned, churn_features)

#disp1 = plot_partial_dependence(rf_tuned, x_smote, [3, 13])

#######################
# XGBoost
########################
xgb = XGBClassifier(random_state=13,
                    use_label_encoder=False,
                    scale_pos_weight=1,
                    learning_rate=0.01,
                    colsample_bytree=0.3,
                    subsample=0.8,
                    n_estimators=1000,
                    reg_alpha=0.3,
                    max_depth=4,
                    gamma=5)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
print('XGB Accuracy Score : ' + str(accuracy_score(y_test, xgb_pred)))
print('XGB Precision Score : ' + str(precision_score(y_test, xgb_pred)))
print('XGB Recall Score : ' + str(recall_score(y_test, xgb_pred)))
print('XGB F1 Score : ' + str(f1_score(y_test, xgb_pred)))

# xgb_feature_importance = create_feature_importance_plot(xgb, churn_features)
#######################
# KNN
########################
# FINDING OPTIMAL K TO AVOID UNDER / OVER FITTING
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 8)
test_recall = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    # Compute accuracy on the testing set
    knn_test_pred = knn.predict(X_test)
    test_recall[i] = recall_score(y_test, knn_test_pred)

# Generate plot
# plt.title('k-NN: Varying Number of Neighbors')
# plt.plot(neighbors, test_recall, label='Testing Recall')
# plt.legend()
# plt.xlabel('Number of Neighbors')
# plt.ylabel('Recall')
# plt.show()

knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the data
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)
print('KNN Accuracy Score : ' + str(accuracy_score(y_test, knn_pred)))
print('KNN Precision Score : ' + str(precision_score(y_test, knn_pred)))
print('KNN Recall Score : ' + str(recall_score(y_test, knn_pred)))
print('KNN F1 Score : ' + str(f1_score(y_test, knn_pred)))
y_prob = knn.predict_proba(X_test)[:, 1]
create_confusion_matrix(y_test, xgb_pred)


########################
# RESTAURANT DATA
########################
# As the churn dataset had few missing values and few duplicates, I will be using the restaurants data from
# kaggle to clean data, display knowledge of regex and merge dataframes
restaurant_1 = import_data("Future50.csv")
restaurant_2 = import_data("Top250.csv")
restaurant_3 = import_data("Independence100.csv")
