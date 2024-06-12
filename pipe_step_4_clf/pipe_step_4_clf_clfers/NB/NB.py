import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from scipy import interp

# Set paths
input_folder_path = '.'
result_auc_csv_path = '../../result_auc_for_each_position/ComplementNB_AUC.csv'
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

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, title, filename, is_multiclass=False):
    plt.figure()
    if is_multiclass:
        if "micro" in fpr and "macro" in fpr:
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["micro"]))
            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]))
    
    for i, color in zip(range(len(fpr)), plt.cm.rainbow(np.linspace(0, 1, len(fpr)))):
        if i in fpr:
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()

# Process each .csv file in the input folder
for filename in os.listdir(input_folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder_path, filename)
        df = pd.read_csv(file_path)

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        n_classes = len(np.unique(y))

        # Binarize the output for multi-class classification
        if n_classes > 2:
            y_bin = label_binarize(y, classes=np.arange(n_classes))
        else:
            y_bin = y

        # Prepare structures to store AUC values
        auc_vs_features = []

        # Perform feature selection and classification for different number of features
        for k in range(1, X.shape[1] + 1):
            X_new = SelectKBest(f_classif, k=k).fit_transform(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X_new, y_bin, test_size=0.3, random_state=42)

            clf = ComplementNB()
            if n_classes > 2:
                classifier = OneVsRestClassifier(clf)
                y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(n_classes):
                    if np.sum(y_test[:, i]) > 0:  # Check if there are positive samples in y_test
                        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])

                # Compute micro-average ROC curve and ROC area
                if np.sum(y_test.ravel()) > 0:
                    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                # Compute macro-average ROC curve and ROC area
                if fpr:
                    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes) if i in fpr]))
                    mean_tpr = np.zeros_like(all_fpr)
                    for i in range(n_classes):
                        if i in fpr:
                            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
                    mean_tpr /= n_classes
                    fpr["macro"] = all_fpr
                    tpr["macro"] = mean_tpr
                    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
                else:
                    fpr["macro"] = np.array([])
                    tpr["macro"] = np.array([])
                    roc_auc["macro"] = 0

                auc_vs_features.append((k, roc_auc.get("micro", 0), roc_auc.get("macro", 0)))
            else:
                if np.sum(y_test) > 0:  # Check if there are positive samples in y_test
                    y_score = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
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

        clf = ComplementNB()
        if n_classes > 2:
            classifier = OneVsRestClassifier(clf)
            y_score_optimal = classifier.fit(X_train_optimal, y_train_optimal).predict_proba(X_test_optimal)

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                if np.sum(y_test_optimal[:, i]) > 0:  # Check if there are positive samples in y_test_optimal
                    fpr[i], tpr[i], _ = roc_curve(y_test_optimal[:, i], y_score_optimal[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

            if np.sum(y_test_optimal.ravel()) > 0:
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test_optimal.ravel(), y_score_optimal.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            if fpr:
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes) if i in fpr]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(n_classes):
                    if i in fpr:
                        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= n_classes
                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            else:
                fpr["macro"] = np.array([])
                tpr["macro"] = np.array([])
                roc_auc["macro"] = 0

            plot_roc_curve(fpr, tpr, roc_auc, f'ROC Curve with Optimal Features for {filename}', 
                           f'../../result_classifier_evaluations/ROC_NB_Optimal_Features_{filename}.png', is_multiclass=True)
        else:
            if np.sum(y_test_optimal) > 0:  # Check if there are positive samples in y_test_optimal
                y_score_optimal = clf.fit(X_train_optimal, y_train_optimal).predict_proba(X_test_optimal)[:, 1]
                fpr, tpr, _ = roc_curve(y_test_optimal, y_score_optimal)
                roc_auc = auc(fpr, tpr)

                plot_roc_curve([fpr], [tpr], [roc_auc], f'ROC Curve with Optimal Features for {filename}', 
                               f'../../result_classifier_evaluations/ROC_NB_Optimal_Features_{filename}.png')

# Save AUC vs Number of Features to CSV
auc_df = pd.DataFrame(auc_data)
auc_df.to_csv(result_auc_csv_path, index=False)

# Plot AUC vs Number of Features
plt.figure()
for filename, auc_vs_features in auc_data.items():
    if len(auc_vs_features[0]) == 3:  # Multiclass
        auc_vs_features = np.array(auc_vs_features)
        plt.plot(auc_vs_features[:, 0], auc_vs_features[:, 1], label=f'{filename} - Micro')
        plt.plot(auc_vs_features[:, 0], auc_vs_features[:, 2], label=f'{filename} - Macro')
    else:
        auc_vs_features = np.array(auc_vs_features)
        plt.plot(auc_vs_features[:, 0], auc_vs_features[:, 1], label=filename)

plt.xlabel('Number of Features')
plt.ylabel('AUC')
plt.title('AUC vs Number of Features')
plt.legend(loc='best')
plt.savefig(result_auc_plot_path)
plt.close()

# Plot ROC curve for all features for all files
plt.figure()
for filename in os.listdir(input_folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder_path, filename)
        df = pd.read_csv(file_path)

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        n_classes = len(np.unique(y))

        # Binarize the output for multi-class classification
        if n_classes > 2:
            y_bin = label_binarize(y, classes=np.arange(n_classes))
        else:
            y_bin = y

        X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42)

        clf = ComplementNB()
        if n_classes > 2:
            classifier = OneVsRestClassifier(clf)
            y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                if np.sum(y_test[:, i]) > 0:  # Check if there are positive samples in y_test
                    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

            if np.sum(y_test.ravel()) > 0:
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            if fpr:
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes) if i in fpr]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(n_classes):
                    if i in fpr:
                        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= n_classes
                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

                plot_roc_curve(fpr, tpr, roc_auc, f'ROC Curve for All Features of {filename}', 
                               f'../../result_classifier_evaluations/ROC_NB_All_Features_{filename}.png', is_multiclass=True)
        else:
            if np.sum(y_test) > 0:  # Check if there are positive samples in y_test
                y_score = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)

                plot_roc_curve([fpr], [tpr], [roc_auc], f'ROC Curve for All Features of {filename}', 
                               f'../../result_classifier_evaluations/ROC_NB_All_Features_{filename}.png')
