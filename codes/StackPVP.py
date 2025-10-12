import os
from sklearn.model_selection import KFold
import numpy as np
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.base import clone

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

def get_oof(clf, n_folds, X_train, y_train, X_test):
    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_train_proba = np.zeros(ntrain)
    oof_test_proba = np.zeros(ntest)
    
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        kf_X_train = X_train[train_index]
        kf_y_train = y_train[train_index]
        kf_X_test = X_train[test_index]
        
        clf.fit(kf_X_train, kf_y_train)
        oof_train_proba[test_index] = clf.predict_proba(kf_X_test)[:, 1]
        oof_test_proba += clf.predict_proba(X_test)[:, 1]
    
    oof_test_proba = oof_test_proba / float(n_folds)
    
    return oof_train_proba.reshape(-1, 1), oof_test_proba.reshape(-1, 1)

def SelectModel(modelname):
    if modelname == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=9, weights='distance', p=2)
    elif modelname == "RF":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=500, max_depth=20, 
                                     min_samples_split=2, min_samples_leaf=1, 
                                     max_features='sqrt',random_state=42, class_weight='balanced')
    elif modelname == "XGboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.01,
                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                            use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1)
    elif modelname == "GBDT":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01,
                                         max_depth=7, subsample=0.8, random_state=42)
    elif modelname == "EF":
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(n_estimators=500, max_depth=15, 
                                   min_samples_split=2, random_state=42, class_weight='balanced')
    elif modelname == "LightGBM":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=500, max_depth=8, learning_rate=0.01,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             class_weight='balanced')
    elif modelname == "ANN":
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                            solver='adam', alpha=0.0001, random_state=42, max_iter=2000,
                            early_stopping=True, validation_fraction=0.1)
    elif modelname == "SVM":
        from sklearn import svm
        model = svm.SVC(probability=True, C=0.7, gamma='auto', kernel='rbf', 
                        random_state=42, class_weight='balanced')
    elif modelname == "DT":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(max_depth=12, min_samples_split=3, 
                                     min_samples_leaf=1, random_state=42,
                                     class_weight='balanced')
    elif modelname == "MLP":
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='tanh',
                            solver='adam', alpha=0.0001, random_state=42, max_iter=2000)
    elif modelname == "AdaBoost":
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(n_estimators=500, learning_rate=0.01, random_state=42)
    elif modelname == "CatBoost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=500, depth=8, learning_rate=0.01,
                                 verbose=0, random_state=42, auto_class_weights='Balanced')
    else:
        raise ValueError(f"Unknown model name: {modelname}")
    return model

def evaluate_model(y_true, y_pred, probas):
    accuracy = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, probas[:, 1])
    confusion = confusion_matrix(y_true, y_pred)
    
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    mcc_numerator = (TP * TN) - (FP * FN)
    mcc_denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0
    
    f1_score = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'auc_score': auc_score,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'mcc': mcc,
        'f1_score': f1_score,
        'confusion_matrix': confusion
    }

def save_evaluation_results(results, filename, algorithm_name, dataset_name, test_set_name):
    """保存评估结果到CSV文件"""
    df = pd.DataFrame({
        'Algorithm': [algorithm_name],
        'Dataset': [dataset_name],
        'Test_Set': [test_set_name],
        'Accuracy': [results['accuracy']],
        'AUC': [results['auc_score']],
        'Sensitivity': [results['sensitivity']],
        'Specificity': [results['specificity']],
        'Precision': [results['precision']],
        'MCC': [results['mcc']],
        'F1_Score': [results['f1_score']]
    })
    
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

def plot_single_roc_curve(y_true, probas, algorithm_name, dataset_name, test_set_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, probas[:, 1])
    auc_score = roc_auc_score(y_true, probas[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {algorithm_name} on {dataset_name} ({test_set_name})')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    filename = f"{algorithm_name}_{dataset_name}_{test_set_name}_ROC.tiff".replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(save_dir, filename), format='tiff', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fpr, tpr, auc_score

def plot_combined_roc_curves(roc_data_dict, algorithm_name, test_set_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    colors = plt.cm.Set3(np.linspace(0, 1, len(roc_data_dict)))
    
    for i, (dataset_name, (fpr, tpr, auc_score)) in enumerate(roc_data_dict.items()):
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{dataset_name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {algorithm_name} on {test_set_name}', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    filename = f"{algorithm_name}_{test_set_name}_Combined_ROC.tiff".replace(" ", "_")
    plt.savefig(os.path.join(save_dir, filename), format='tiff', dpi=300, bbox_inches='tight')
    plt.close()

def run_stacking_for_combination(X_train_list, X_test_list, y_train, y_test, dataset_names, 
                               combination_name, second_level_models):
    print(f"\nRunning stacking for combination: {combination_name}")
    
    newtrfeature_list = []
    newtefeature_list = []
    base_models = []
    
    for data_norm, test_norm, data_name in zip(X_train_list, X_test_list, dataset_names):
        print(f"Processing {data_name} features for stacking...")
        for modelname in modelist:
            try:
                clf = SelectModel(modelname)
                oof_train, oof_test = get_oof(clf=clf, n_folds=10,  
                                             X_train=data_norm, y_train=y_train, 
                                             X_test=test_norm)
                
                newtrfeature_list.append(oof_train)
                newtefeature_list.append(oof_test)
                base_models.append(clf)
            except Exception as e:
                print(f"    Error with {modelname}: {e}")
                zero_train = np.zeros((len(y_train), 1))
                zero_test = np.zeros((len(y_test), 1))
                newtrfeature_list.append(zero_train)
                newtefeature_list.append(zero_test)
                base_models.append(None)
    
    newtrfeature = np.concatenate(newtrfeature_list, axis=1)
    newtefeature = np.concatenate(newtefeature_list, axis=1)
    
    print(f"Meta-feature shapes: Train {newtrfeature.shape}, Test {newtefeature.shape}")
    
    results_dict = {}
    roc_data_dict = {}
    
    for algo_name, model in second_level_models.items():
        try:
            model.fit(newtrfeature, y_train)
            pred_test = model.predict(newtefeature)
            probas_test = model.predict_proba(newtefeature)
            
            results_test = evaluate_model(y_test, pred_test, probas_test)
            results_dict[algo_name] = results_test
            
            fpr, tpr, auc_score = plot_single_roc_curve(
                y_test, probas_test, algo_name, combination_name, test_set_name, 
                "results/individual_roc_curves"
            )
            roc_data_dict[algo_name] = (fpr, tpr, auc_score)
            
            print(f"{algo_name} on {combination_name}: AUC = {results_test['auc_score']:.4f}")
            
        except Exception as e:
            print(f"Error with {algo_name} on {combination_name}: {e}")
    
    return results_dict, roc_data_dict

print("Loading data with correct sample sizes...")

os.makedirs("results", exist_ok=True)
os.makedirs("results/individual_roc_curves", exist_ok=True)
os.makedirs("results/combined_roc_curves", exist_ok=True)

x1 = np.loadtxt("./PSSM_train_and_test/PSSM.F_train.csv", delimiter=",")
x2 = np.loadtxt("./PSSM_train_and_test/PSSM.Var_train8.csv", delimiter=",")
x3 = np.loadtxt("./PSSM_train_and_test/PSSM.RFECV_train.csv", delimiter=",")

test_x1_1 = np.loadtxt("./PSSM_train_and_test/PSSM.F_test1.csv", delimiter=",")
test_x2_1 = np.loadtxt("./PSSM_train_and_test/PSSM.Var_test1.csv", delimiter=",")
test_x3_1 = np.loadtxt("./PSSM_train_and_test/PSSM.RFECV_test1.csv", delimiter=",")

test_x1_2 = np.loadtxt("./PSSM_train_and_test/PSSM.F_test2.csv", delimiter=",")
test_x2_2 = np.loadtxt("./PSSM_train_and_test/PSSM.Var_test2.csv", delimiter=",")
test_x3_2 = np.loadtxt("./PSSM_train_and_test/PSSM.RFECV_test2.csv", delimiter=",")

y_train = np.array([1] * 349 + [0] * 458)
y_test_1 = np.array([1] * 63 + [0] * 63)
y_test_2 = np.array([1] * 30 + [0] * 64)

scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()

x1_norm = scaler1.fit_transform(x1)
x2_norm = scaler2.fit_transform(x2)
x3_norm = scaler3.fit_transform(x3)

test_x1_norm_1 = scaler1.transform(test_x1_1)
test_x2_norm_1 = scaler2.transform(test_x2_1)
test_x3_norm_1 = scaler3.transform(test_x3_1)

test_x1_norm_2 = scaler1.transform(test_x1_2)
test_x2_norm_2 = scaler2.transform(test_x2_2)
test_x3_norm_2 = scaler3.transform(test_x3_2)

modelist = ['KNN', 'RF', 'XGboost', 'GBDT', 'EF', 'LightGBM', 'ANN', 'SVM', 'DT', 'MLP', 'AdaBoost', 'CatBoost']
print(f"\nUsing {len(modelist)} base models: {modelist}")

second_level_models = {
    'LR': LogisticRegression(random_state=42, max_iter=2000, C=0.1),
    'RF': RandomForestClassifier(n_estimators=200, random_state=42),
    'SVM': SVC(probability=True, C=1.0, kernel='rbf', random_state=42)
}

dataset_combinations = {
    'F_only': [(x1_norm, test_x1_norm_1, test_x1_norm_2), ['F']],
    'Var_only': [(x2_norm, test_x2_norm_1, test_x2_norm_2), ['Var']],
    'RFECV_only': [(x3_norm, test_x3_norm_1, test_x3_norm_2), ['RFECV']],
    'F_Var': [(x1_norm, test_x1_norm_1, test_x1_norm_2), 
              (x2_norm, test_x2_norm_1, test_x2_norm_2), ['F', 'Var']],
    'F_RFECV': [(x1_norm, test_x1_norm_1, test_x1_norm_2), 
                (x3_norm, test_x3_norm_1, test_x3_norm_2), ['F', 'RFECV']],
    'Var_RFECV': [(x2_norm, test_x2_norm_1, test_x2_norm_2), 
                  (x3_norm, test_x3_norm_1, test_x3_norm_2), ['Var', 'RFECV']],
    'All_three': [(x1_norm, test_x1_norm_1, test_x1_norm_2), 
                  (x2_norm, test_x2_norm_1, test_x2_norm_2), 
                  (x3_norm, test_x3_norm_1, test_x3_norm_2), ['F', 'Var', 'RFECV']]
}

for test_set_name, y_test, test_suffix in [('Train_CV', y_train, '_train'), 
                                         ('Test1', y_test_1, '_test1'), 
                                         ('Test2', y_test_2, '_test2')]:
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {test_set_name}")
    print(f"{'='*60}")
    
    algorithm_roc_data = {algo: {} for algo in second_level_models.keys()}
    all_results = []
    
    for combo_name, combo_data in dataset_combinations.items():
        X_train_list = []
        X_test_list = []
        dataset_names = combo_data[-1]
        
        for i, data_tuple in enumerate(combo_data[:-1]):
            X_train_list.append(data_tuple[0])
            if test_set_name == 'Test1':
                X_test_list.append(data_tuple[1])
            elif test_set_name == 'Test2':
                X_test_list.append(data_tuple[2])
            else:  
                X_test_list.append(data_tuple[0])  
        
        results_dict, roc_data_dict = run_stacking_for_combination(
            X_train_list, X_test_list, y_train, y_test, dataset_names, 
            combo_name, second_level_models
        )
        
        for algo_name, results in results_dict.items():
            save_evaluation_results(results, f"results/evaluation_results{test_suffix}.csv", 
                                  algo_name, combo_name, test_set_name)
            
            if algo_name in roc_data_dict:
                algorithm_roc_data[algo_name][combo_name] = roc_data_dict[algo_name]
            
            all_results.append({
                'Algorithm': algo_name,
                'Dataset': combo_name,
                'Test_Set': test_set_name,
                'Accuracy': results['accuracy'],
                'AUC': results['auc_score'],
                'Sensitivity': results['sensitivity'],
                'Specificity': results['specificity'],
                'Precision': results['precision'],
                'MCC': results['mcc'],
                'F1_Score': results['f1_score']
            })
    
    for algo_name, roc_data in algorithm_roc_data.items():
        if roc_data:  
            plot_combined_roc_curves(roc_data, algo_name, test_set_name, 
                                   "results/combined_roc_curves")
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(f"results/summary_results{test_suffix}.csv", index=False)
        print(f"Summary saved for {test_set_name}")

print("\nAll stacking combinations completed!")
print("Individual ROC curves saved to 'results/individual_roc_curves/'")
print("Combined ROC curves saved to 'results/combined_roc_curves/'")
print("Evaluation results saved to 'results/evaluation_results*.csv'")
print("Summary results saved to 'results/summary_results*.csv'")