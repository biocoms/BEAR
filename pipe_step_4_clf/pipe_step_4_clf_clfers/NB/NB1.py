import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def process_files(folder_path):
    results = []
    all_roc_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            # Determine if the problem is binary or multiclass
            classes = np.unique(y)
            is_multiclass = len(classes) > 2

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            if is_multiclass:
                y_test_binarized = label_binarize(y_test, classes=classes)
                clf = OneVsRestClassifier(GaussianNB())
            else:
                clf = GaussianNB()

            clf.fit(X_train, y_train)
            y_score = clf.predict_proba(X_test)

            # Calculate ROC AUC for each feature count
            auc_scores = []
            for i in range(1, X_train.shape[1] + 1):
                clf.fit(X_train.iloc[:, :i], y_train)
                if is_multiclass:
                    y_score = clf.predict_proba(X_test.iloc[:, :i])
                    roc_auc = roc_auc_score(y_test_binarized, y_score, multi_class='ovr', average='macro')
                    auc_scores.append(roc_auc)
                else:
                    y_score = clf.predict_proba(X_test.iloc[:, :i])[:, 1]
                    roc_auc = roc_auc_score(y_test, y_score)
                    auc_scores.append(roc_auc)

            results.append({
                'file': filename,
                'auc_scores': auc_scores
            })

            # Calculate ROC curves
            if is_multiclass:
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(len(classes)):
                    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                all_roc_data.append({'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc, 'classes': classes})
            else:
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                all_roc_data.append({'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc})

    # Save results to CSV
    auc_df = pd.DataFrame({
        result['file']: result['auc_scores'] for result in results
    })
    auc_df.to_csv('auc_vs_features.csv', index=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    for result in results:
        plt.plot(result['auc_scores'], label=f"{result['file']} (AUC)")
    plt.title('AUC vs Number of Features')
    plt.xlabel('Number of Features')
    plt.ylabel('AUC Score')
    plt.legend()
    plt.savefig('auc_vs_features_plot.png')

    # Plot ROC curves for each file
    plt.figure(figsize=(10, 6))
    for roc_data in all_roc_data:
        if 'classes' in roc_data:
            for i in roc_data['classes']:
                plt.plot(roc_data['fpr'][i], roc_data['tpr'][i], label=f'Class {i} AUC = {roc_data["roc_auc"][i]:.2f}')
        else:
            plt.plot(roc_data['fpr'], roc_data['tpr'], label=f'ROC AUC = {roc_data["roc_auc"]:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('roc_curve.png')

# Usage
process_files('.')
