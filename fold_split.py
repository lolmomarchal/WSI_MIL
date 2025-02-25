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
    
    folds = [[] for _ in range(n_splits)]  

    for subtype, subtype_data in data.groupby(cancer_subtype_column):
        # Group by patient and label counts
        grouped = subtype_data.groupby(patient_id_column)[label_column].value_counts().unstack(fill_value=0).reset_index()

        # Ensure label columns exist
        if 0 not in grouped.columns:
            grouped[0] = 0
        if 1 not in grouped.columns:
            grouped[1] = 0

        grouped.columns = [patient_id_column, 'label_0_count', 'label_1_count']
        
        grouped = grouped.sample(frac=1, random_state=random_state)

        patients_label_0 = grouped[grouped["label_0_count"] > 0][patient_id_column].tolist()
        patients_label_1 = grouped[grouped["label_1_count"] > 0][patient_id_column].tolist()

        np.random.shuffle(patients_label_0)
        np.random.shuffle(patients_label_1)

        for i, pid in enumerate(patients_label_0):
            folds[i % n_splits].append(pid)
        for i, pid in enumerate(patients_label_1):
            folds[i % n_splits].append(pid)

    fold_splits = []
    for i in range(n_splits):
        val_patients = set(folds[i])
        val_data = data[data[patient_id_column].isin(val_patients)]
        train_data = data[~data[patient_id_column].isin(val_patients)]
        fold_splits.append((train_data, val_data))

    return fold_splits
