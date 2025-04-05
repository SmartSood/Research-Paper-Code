# %%
import pandas as pd
import os
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import plotly.express as px

# Load and process your data from the 'D:\CSV_files' directory
directory = '/Users/smarthsood/Desktop/CSV_files'
autism = []
td = []

# Load and process data
for filename in os.listdir(directory):
    if filename.split(".")[0].endswith("1"):
        f = os.path.join(directory, filename)
        df = pd.read_csv(f, header=None)
        for i in range(0, 6):
            frame = df.iloc[:, 30 * i:30 * i + 30]
            result = np.asarray(frame.T.corr())
            a = list(result[np.triu_indices(result.shape[1], k=1)])
            autism.append([])
            autism[-1].extend(a)
    else:
        f = os.path.join(directory, filename)
        df = pd.read_csv(f, header=None)
        for i in range(0, 6):
            frame = df.iloc[:, 30 * i:30 * i + 30]
            result = np.asarray(frame.T.corr())
            a = list(result[np.triu_indices(result.shape[1], k=1)])
            td.append([])
            td[-1].extend(a)
# Combine all data into a single array
X = np.vstack((autism, td))
y = np.array([1] * len(autism) + [0] * len(td))  # Label 1 for autism, 0 for TD


# %%

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifiers
classifiers = {
    "SVM Linear": svm.SVC(kernel='linear'),
    "SVM Sigmoid": svm.SVC(kernel='sigmoid'),
    "SVM RBF": svm.SVC(kernel='rbf'),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": xgb.XGBClassifier(),
    "MLP Classifier": MLPClassifier()
}

# Perform feature ranking using XGBoost on the entire dataset
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train, y_train)
feature_importances = xgb_classifier.feature_importances_

# Lists to store accuracy, sensitivity, specificity, precision, F1 score, and feature count for each model
accuracy_dict = {classifier_name: [] for classifier_name in classifiers.keys()}
sensitivity_dict = {classifier_name: [] for classifier_name in classifiers.keys()}
specificity_dict = {classifier_name: [] for classifier_name in classifiers.keys()}
precision_dict = {classifier_name: [] for classifier_name in classifiers.keys()}
f1_score_dict = {classifier_name: [] for classifier_name in classifiers.keys()}
feature_count_list = []


# %%

# Start with the top 100 features and increment by 100
max_features = min(X.shape[1], 27700)
for num_features in range(100, max_features + 1, 100):
    feature_count_list.append(num_features)

    for classifier_name, classifier in classifiers.items():
        # Implement 5-fold accuracy, sensitivity, specificity, precision, and F1 score calculation
        kf = KFold(n_splits=5)
        accuracies = []
        sensitivities = []
        specificities = []
        precisions = []
        f1_scores = []

        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            # Select the top k features based on XGBoost feature rankings
            top_feature_indices = np.argsort(feature_importances)[-num_features:]
            X_train_selected = X_train_fold[:, top_feature_indices]
            X_val_selected = X_val_fold[:, top_feature_indices]

            # Train the classifier on the selected features
            classifier.fit(X_train_selected, y_train_fold)

            # Calculate predictions
            y_pred = classifier.predict(X_val_selected)

            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_val_fold, y_pred).ravel()

            # Calculate sensitivity, specificity, precision, and F1 score
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            precision = tp / (tp + fp)
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

            accuracies.append(accuracy_score(y_val_fold, y_pred))
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            precisions.append(precision)
            f1_scores.append(f1)

        # Calculate the average metrics for the current model and number of features
        average_accuracy = np.mean(accuracies)
        average_sensitivity = np.mean(sensitivities)
        average_specificity = np.mean(specificities)
        average_precision = np.mean(precisions)
        average_f1_score = np.mean(f1_scores)

        accuracy_dict[classifier_name].append(average_accuracy)
        sensitivity_dict[classifier_name].append(average_sensitivity)
        specificity_dict[classifier_name].append(average_specificity)
        precision_dict[classifier_name].append(average_precision)
        f1_score_dict[classifier_name].append(average_f1_score)

        print(f'5-Fold Metrics for {classifier_name} with {num_features} features:')
        print(f'Accuracy: {average_accuracy * 100:.2f}%')
        print(f'Sensitivity: {average_sensitivity * 100:.2f}%')
        print(f'Specificity: {average_specificity * 100:.2f}%')
        print(f'Precision: {average_precision * 100:.2f}%')
        print(f'F1 Score: {average_f1_score * 100:.2f}%')

# Create interactive plots using Plotly Express for each classifier and each metric


# %%

metrics_to_plot = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score']
metric_dicts = [accuracy_dict, sensitivity_dict, specificity_dict, precision_dict, f1_score_dict]

for metric_name, metric_dict in zip(metrics_to_plot, metric_dicts):
    for classifier_name, metric_list in metric_dict.items():
                        fig = px.line(x=feature_count_list, y=metric_list, labels={"x": "Number of Top Features", "y": metric_name}, 
                        title=f"{metric_name} Plot for {classifier_name}")
        
                        fig.update_layout(legend_title_text="Classifier")
                        
                        # Generate file names for image and HTML files
                        image_file_name = f"{metric_name.lower().replace(' ', '_')}_plot_{classifier_name}.jpg"
                        html_file_name = f"{metric_name.lower().replace(' ', '_')}_plot_{classifier_name}_interactive.html"
                        
                        # Save the interactive plot as a JPG image and an HTML file
                        fig.write_image(image_file_name)
                        fig.write_html(html_file_name)



