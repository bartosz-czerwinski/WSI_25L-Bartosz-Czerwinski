import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score

from cwiczenie4.DecisionTree import DecisionTreeClassifier
from cwiczenie4.RandomForest import RandomForestClassifier

pd.set_option('display.max_columns', None)
column_names = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'HeartDisease', 'AgeGroup', 'RestingBP_Category', 'Cholesterol_Category', 'MaxHR_Category', 'Oldpeak_Category']
df = pd.read_csv("lab-4-dataset.csv", skiprows=1, header=None, names=column_names)

df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col], _ = pd.factorize(df_encoded[col])

X = df_encoded.drop(columns=["HeartDisease"]).to_numpy()
y = df_encoded["HeartDisease"].to_numpy()

# Parametry testów
n_estimators_list = [1, 5, 10, 20, 50, 100, 250, 500, 1000]
n_runs_per_setting = 10


# Wyniki
avg_accuracies = []
avg_precisions = []
avg_recalls = []
avg_aucs = []
roc_curves = []


test_sets = []

for i in range(n_runs_per_setting):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    test_sets.append((X_train, X_test, y_train, y_test))


aucs_plot = []
for n_trees in n_estimators_list:
    accuracies, precisions, recalls, aucs = [], [], [], []
    best_auc = 0
    for i in range(n_runs_per_setting):
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_test, y_train, y_test = test_sets[i]

        forest = RandomForestClassifier(n_estimators=n_trees, max_depth=10)
        forest.fit(X_train, y_train)
        preds = forest.predict(X_test)


        probs = np.mean([tree.predict(X_test[:, idxs]) for tree, idxs in zip(forest.trees, forest.feature_indices_list)], axis=0)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)


        if i == 0:
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_curves.append((fpr, tpr))
            best_auc = auc

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        aucs.append(auc)



    # Średnie metryki
    avg_accuracies.append(np.mean(accuracies))
    avg_precisions.append(np.mean(precisions))
    avg_recalls.append(np.mean(recalls))
    avg_aucs.append(np.mean(aucs))
    aucs_plot.append(best_auc)

    print(f"n_estimators = {n_trees:>3}: acc={np.mean(accuracies):.3f}, prec={np.mean(precisions):.3f}, "
          f"rec={np.mean(recalls):.3f}, auc={np.mean(aucs):.3f}")

# Wykresy metryk
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_list, avg_accuracies, label='Dokładność')
plt.plot(n_estimators_list, avg_precisions, label='Precyzja')
plt.plot(n_estimators_list, avg_recalls, label='Czułość')
plt.plot(n_estimators_list, avg_aucs, label='AUC')
plt.xlabel("Liczba drzew")
plt.ylabel("Średnia wartość metryki")
plt.title("Metryki klasyfikacji vs liczba drzew")
plt.legend()
plt.grid(True)
plt.savefig("metrics_vs_n_estimators.png")
plt.show()
plt.close()

for i in range(len(n_estimators_list)):
    fpr, tpr = roc_curves[i]
    auc = aucs_plot[i]
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'Krzywa ROC (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Krzywa ROC - Las losowy o liczbie drzew: {n_estimators_list[i]}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'roc_curve_{n_estimators_list[i]}_trees.png')
    plt.close()
