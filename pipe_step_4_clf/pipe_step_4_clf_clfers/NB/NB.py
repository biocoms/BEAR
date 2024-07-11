import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt

# Set paths
input_folder_path = '.'
result_auc_csv_path = '../../result_auc_for_each_position/NB_AUC.csv'
result_auc_plot_path = '../../result_classifier_evaluations/NB_AUC_vs_Features.png'
roc_all_features_path = '../../result_classifier_evaluations/ROC_NB_All_Features.png'
roc_optimal_features_path = '../../result_classifier_evaluations/ROC_NB_Optimal_Features.png'

# Ensure output directories exist
os.makedirs(os.path.dirname(result_auc_csv_path), exist_ok=True)
os.makedirs(os.path.dirname(result_auc_plot_path), exist_ok=True)
os.makedirs(os.path.dirname(roc_all_features_path), exist_ok=True)
os.makedirs(os.path.dirname(roc_optimal_features_path), exist_ok=True)

# Initialize data structures
auc_data = {}
all_fpr_all_features = {}
all_tpr_all_features = {}
all_roc_auc_all_features = {}
optimal_fpr_all_features = {}
optimal_tpr_all_features = {}
optimal_roc_auc_all_features = {}

# Function to plot combined ROC curve
def plot_combined_roc_curve(fpr_dict, tpr_dict, roc_auc_dict, title, filename):
    plt.figure(figsize=(10, 8))
    for key in fpr_dict:
        if isinstance(fpr_dict[key], dict):  # Multi-class
            for sub_key in fpr_dict[key]:
                plt.plot(fpr_dict[key][sub_key], tpr_dict[key][sub_key],
                         label=f'{key} - {sub_key} (area = {roc_auc_dict[key][sub_key]:0.2f})')
        else:  # Binary
            fpr, tpr, roc_auc = fpr_dict[key]
            plt.plot(fpr, tpr,
                     label=f'{key} (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Process each .csv file in the input folder
for filename in os.listdir(input_folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder_path, filename)
        df = pd.read_csv(file_path)

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Determine if binary or multi-class
        n_classes = len(np.unique(y))
        if n_classes > 2:
            y_bin = label_binarize(y, classes=np.unique(y))
        else:
            y_bin = (y == 'Positive').astype(int)  # Convert 'Negative' to 0 and 'Positive' to 1

        # Scale the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Prepare structures to store AUC values
        auc_vs_features = []

        # Perform feature selection and classification for different number of features
        for k in range(1, X.shape[1] + 1):
            X_new = SelectKBest(f_classif, k=k).fit_transform(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X_new, y_bin, test_size=0.3, random_state=42)

            clf = GaussianNB()
            if n_classes > 2:
                clf = OneVsRestClassifier(clf)
                y_score = clf.fit(X_train, y_train).predict_proba(X_test)
            else:
                y_score = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            if n_classes > 2:
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(n_classes):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= n_classes
                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
                auc_vs_features.append((k, roc_auc["micro"], roc_auc["macro"]))
            else:
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                auc_vs_features.append((k, roc_auc))

        # Store AUC vs Features data
        auc_data[filename] = auc_vs_features

        # Find the optimal number of features based on AUC
        if n_classes > 2:
            optimal_features = max(auc_vs_features, key=lambda x: x[1])[0]  # Using micro average AUC
        else:
            optimal_features = max(auc_vs_features, key=lambda x: x[1])[0]

        # Perform classification with optimal number of features
        X_optimal = SelectKBest(f_classif, k=optimal_features).fit_transform(X, y)
        X_train_optimal, X_test_optimal, y_train_optimal, y_test_optimal = train_test_split(X_optimal, y_bin, test_size=0.3, random_state=42)

        clf = GaussianNB()
        if n_classes > 2:
            clf = OneVsRestClassifier(clf)
            y_score_optimal = clf.fit(X_train_optimal, y_train_optimal).predict_proba(X_test_optimal)
        else:
            y_score_optimal = clf.fit(X_train_optimal, y_train_optimal).predict_proba(X_test_optimal)[:, 1]

        fpr_optimal = dict()
        tpr_optimal = dict()
        roc_auc_optimal = dict()
        if n_classes > 2:
            for i in range(n_classes):
                fpr_optimal[i], tpr_optimal[i], _ = roc_curve(y_test_optimal[:, i], y_score_optimal[:, i])
                roc_auc_optimal[i] = auc(fpr_optimal[i], tpr_optimal[i])
            fpr_optimal["micro"], tpr_optimal["micro"], _ = roc_curve(y_test_optimal.ravel(), y_score_optimal.ravel())
            roc_auc_optimal["micro"] = auc(fpr_optimal["micro"], tpr_optimal["micro"])
            all_fpr_optimal = np.unique(np.concatenate([fpr_optimal[i] for i in range(n_classes)]))
            mean_tpr_optimal = np.zeros_like(all_fpr_optimal)
            for i in range(n_classes):
                mean_tpr_optimal += np.interp(all_fpr_optimal, fpr_optimal[i], tpr_optimal[i])
            mean_tpr_optimal /= n_classes
            fpr_optimal["macro"] = all_fpr_optimal
            tpr_optimal["macro"] = mean_tpr_optimal
            roc_auc_optimal["macro"] = auc(fpr_optimal["macro"], tpr_optimal["macro"])
        else:
            fpr_optimal, tpr_optimal, _ = roc_curve(y_test_optimal, y_score_optimal)
            roc_auc_optimal = auc(fpr_optimal, tpr_optimal)

        # Store the ROC data for plotting
        if n_classes > 2:
            all_fpr_all_features[filename] = fpr
            all_tpr_all_features[filename] = tpr
            all_roc_auc_all_features[filename] = roc_auc
            optimal_fpr_all_features[filename] = fpr_optimal
            optimal_tpr_all_features[filename] = tpr_optimal
            optimal_roc_auc_all_features[filename] = roc_auc_optimal
        else:
            all_fpr_all_features[filename] = (fpr, tpr, roc_auc)
            optimal_fpr_all_features[filename] = (fpr_optimal, tpr_optimal, roc_auc_optimal)

# Save AUC vs Number of Features to CSV
auc_df = pd.DataFrame(auc_data)
auc_df.to_csv(result_auc_csv_path, index=False)

# Plot AUC vs Number of Features
plt.figure(figsize=(10, 8))
for filename, auc_vs_features in auc_data.items():
    auc_vs_features = np.array(auc_vs_features)
    plt.plot(auc_vs_features[:, 0], auc_vs_features[:, 1], label=f'{filename} - Micro' if n_classes > 2 else filename)
    if n_classes > 2:
        plt.plot(auc_vs_features[:, 0], auc_vs_features[:, 2], label=f'{filename} - Macro')

plt.xlabel('Number of Features')
plt.ylabel('AUC')
plt.title('AUC vs Number of Features')
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')
plt.tight_layout()
plt.savefig(result_auc_plot_path)
plt.close()

# Plot combined ROC curve for all features
plot_combined_roc_curve(all_fpr_all_features, all_tpr_all_features, all_roc_auc_all_features,
                        'ROC Curve for All Features', roc_all_features_path)

# Plot combined ROC curve for optimal features
plot_combined_roc_curve(optimal_fpr_all_features, optimal_tpr_all_features, optimal_roc_auc_all_features,
                        'ROC Curve for Optimal Features', roc_optimal_features_path)