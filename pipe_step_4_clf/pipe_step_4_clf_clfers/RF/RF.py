import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
from scipy import interp

def roc_curve_multiclass(y_true, y_score, n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_score[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc

def find_optimal_features_and_plot_roc_rf(folder_path):
    all_auc_scores = []
    optimal_num_features = 0
    highest_auc = 0
    
    encoder = LabelEncoder()
    plt.figure(figsize=(10, 8))

    # First pass: find the highest AUC across all features for each file
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            X = df.drop('class', axis=1)
            y = df['class']
            y_encoded = encoder.fit_transform(y)
            num_classes = len(np.unique(y_encoded))

            # Calculate AUC for all features
            cv = StratifiedKFold(n_splits=5)
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            for train, test in cv.split(X, y_encoded):
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X.iloc[train], y_encoded[train])
                y_score = clf.predict_proba(X.iloc[test])

                if num_classes == 2:
                    fpr, tpr, _ = roc_curve(y_encoded[test], y_score[:, 1])
                    roc_auc = auc(fpr, tpr)
                else:
                    fpr, tpr, roc_auc = roc_curve_multiclass(y_encoded[test], y_score, num_classes)

                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(roc_auc if num_classes == 2 else roc_auc["macro"])

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, label=f'{filename} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')

            # Find the optimal number of features for each file
            for i in range(1, X.shape[1] + 1):
                X_temp = X.iloc[:, :i]
                aucs = []
                for train, test in cv.split(X_temp, y_encoded):
                    clf = RandomForestClassifier(n_estimators=100, random_state=42)
                    clf.fit(X_temp.iloc[train], y_encoded[train])
                    y_score = clf.predict_proba(X_temp.iloc[test])
                    roc_auc = roc_auc_score(y_encoded[test], y_score[:, 1]) if num_classes == 2 else roc_auc_score(y_encoded[test], y_score, multi_class='ovr', average='macro')
                    aucs.append(roc_auc)

                mean_auc = np.mean(aucs)
                all_auc_scores.append([filename, i, mean_auc])

                if mean_auc > highest_auc:
                    highest_auc = mean_auc
                    optimal_num_features = i
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves with All Features Across Files (RF)')
    plt.legend(loc="lower right")
    plt.savefig('../../result_classifier_evaluations/ROC_RF_All_Features.png')
    plt.close()

    # Convert all AUC scores into a DataFrame and then pivot
    all_auc_df = pd.DataFrame(all_auc_scores, columns=['Filename', 'Number of Features', 'AUC'])
    pivot_df = all_auc_df.pivot(index='Number of Features', columns='Filename', values='AUC').reset_index()
    pivot_df.to_csv('../../result_auc_for_each_position/ComplementRF_AUC.csv', index=False)

    # Plot AUC vs. Number of Features for each file
    for filename in all_auc_df['Filename'].unique():
        temp_df = all_auc_df[all_auc_df['Filename'] == filename]
        plt.figure()
        plt.plot(temp_df['Number of Features'], temp_df['AUC'], label=f'{filename}')
        plt.title('AUC vs. Number of Features (RF)')
        plt.xlabel('Number of Features')
        plt.ylabel('AUC')
        plt.legend()
        plt.savefig(f'../../result_classifier_evaluations/RF_AUC_vs_Features_{filename}.png')
        plt.close()

    # Second pass: use the optimal number of features to plot ROC curves
    plt.figure(figsize=(10, 8))
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            X = df.iloc[:, :optimal_num_features]  # Use optimal number of features
            y = df['class']
            y_encoded = encoder.fit_transform(y)

            cv = StratifiedKFold(n_splits=5)
            for train, test in cv.split(X, y_encoded):
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X.iloc[train], y_encoded[train])
                y_score = clf.predict_proba(X.iloc[test])
                fpr, tpr, _ = roc_curve(y_encoded[test], y_score[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{filename} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves with Optimal Number of Features Across Files (RF)')
    plt.legend(loc="lower right")
    plt.savefig('../../result_classifier_evaluations/ROC_RF_Optimal_Features.png')
    plt.show()

folder_path = '.'  # Adjust as necessary
find_optimal_features_and_plot_roc_rf(folder_path)
