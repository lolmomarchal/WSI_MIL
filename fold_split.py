import pandas as pd
import numpy as np

def stratified_train_test_split(data, label_column, patient_id_column, cancer_subtype_column = "cancer_subtype", test_size=0.2, random_state=None, split=True):
    if random_state is not None:
        np.random.seed(random_state)
    if not split:
        mask = data.split.str.contains("test", case=False)
        test = data[~mask]
        training = data[mask]
        return test, training
    
    test_patients = []
    
    for subtype, subtype_data in data.groupby(cancer_subtype_column):
        total_label_0 = (subtype_data[label_column] == 0).sum()
        total_label_1 = (subtype_data[label_column] == 1).sum()
        total_samples = len(subtype_data)
        
        target_test_size = int(total_samples * test_size)
        target_test_label_0 = int(target_test_size * (total_label_0 / total_samples))
        target_test_label_1 = int(target_test_size * (total_label_1 / total_samples))
        
        grouped = subtype_data.groupby(patient_id_column)[label_column].value_counts().unstack(fill_value=0).reset_index()
        if 0 not in grouped.columns:
            grouped[0] = 0
        if 1 not in grouped.columns:
            grouped[1] = 0

        grouped = grouped.rename(columns={0: 'label_0_count', 1: 'label_1_count'})
        grouped = grouped.sample(frac=1, random_state=random_state)
        
        test_label_0_count = 0
        test_label_1_count = 0
        
        for _, row in grouped.iterrows():
            if (test_label_0_count < target_test_label_0) or (test_label_1_count < target_test_label_1):
                test_patients.append(row[patient_id_column])
                test_label_0_count += row['label_0_count']
                test_label_1_count += row['label_1_count']
            else:
                break
    
    test_data = data[data[patient_id_column].isin(test_patients)]
    train_data = data[~data[patient_id_column].isin(test_patients)]
    
    return train_data, test_data

def stratified_k_fold_split(data, label_column, patient_id_column, cancer_subtype_column = "cancer_subtype", n_splits=5, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    folds = []
    
    for subtype, subtype_data in data.groupby(cancer_subtype_column):
        grouped = subtype_data.groupby(patient_id_column)[label_column].value_counts().unstack(fill_value=0).reset_index()
        grouped.columns = [patient_id_column, 'label_0_count', 'label_1_count']
        grouped = grouped.sample(frac=1, random_state=random_state)
        
        patient_ids = grouped[patient_id_column].tolist()
        total_samples = len(subtype_data)
        
        target_fold_size = int(total_samples / n_splits)
        target_fold_label_0 = int(target_fold_size * (grouped['label_0_count'].sum() / total_samples))
        target_fold_label_1 = int(target_fold_size * (grouped['label_1_count'].sum() / total_samples))
        
        subtype_folds = []
        
        for _ in range(n_splits):
            fold_patients = []
            fold_label_0_count = 0
            fold_label_1_count = 0
            
            for pid in patient_ids:
                patient_data = grouped[grouped[patient_id_column] == pid]
                if (fold_label_0_count < target_fold_label_0) or (fold_label_1_count < target_fold_label_1):
                    fold_patients.append(pid)
                    fold_label_0_count += patient_data['label_0_count'].values[0]
                    fold_label_1_count += patient_data['label_1_count'].values[0]
                else:
                    break
            
            patient_ids = [pid for pid in patient_ids if pid not in fold_patients]
            val_data = subtype_data[subtype_data[patient_id_column].isin(fold_patients)]
            train_data = subtype_data[~subtype_data[patient_id_column].isin(fold_patients)]
            
            subtype_folds.append((train_data, val_data))
        
        folds.extend(subtype_folds)
    
    return folds
