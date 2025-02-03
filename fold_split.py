import pandas as pd
import numpy as np

def stratified_train_test_split(data, label_column, patient_id_column, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    # Get label distribution at sample level
    total_label_0 = (data[label_column] == 0).sum()
    total_label_1 = (data[label_column] == 1).sum()
    total_samples = len(data)

    target_test_size = int(total_samples * test_size)
    target_test_label_0 = int(target_test_size * (total_label_0 / total_samples))
    target_test_label_1 = int(target_test_size * (total_label_1 / total_samples))

    # Group by patient and get patient-level label distributions
    grouped = data.groupby(patient_id_column)[label_column].value_counts().unstack(fill_value=0).reset_index()
    grouped.columns = [patient_id_column, 'label_0_count', 'label_1_count']

    # Shuffle patients
    grouped = grouped.sample(frac=1, random_state=random_state)

    test_patients = []
    test_label_0_count = 0
    test_label_1_count = 0

    for _, row in grouped.iterrows():
        if (test_label_0_count < target_test_label_0) or (test_label_1_count < target_test_label_1):
            test_patients.append(row[patient_id_column])
            test_label_0_count += row['label_0_count']
            test_label_1_count += row['label_1_count']
        else:
            break

    # Split data
    test_data = data[data[patient_id_column].isin(test_patients)]
    train_data = data[~data[patient_id_column].isin(test_patients)]

    return train_data, test_data


def stratified_k_fold_split(data, label_column, patient_id_column, n_splits=5, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    # Get label distribution
    total_label_0 = (data[label_column] == 0).sum()
    total_label_1 = (data[label_column] == 1).sum()
    total_samples = len(data)

    target_fold_size = int(total_samples / n_splits)
    target_fold_label_0 = int(target_fold_size * (total_label_0 / total_samples))
    target_fold_label_1 = int(target_fold_size * (total_label_1 / total_samples))

    # Group by patient and count labels per patient
    grouped = data.groupby(patient_id_column)[label_column].value_counts().unstack(fill_value=0).reset_index()
    grouped.columns = [patient_id_column, 'label_0_count', 'label_1_count']

    # Shuffle patients
    grouped = grouped.sample(frac=1, random_state=random_state)

    folds = []
    patient_ids = grouped[patient_id_column].tolist()
    
    for i in range(n_splits):
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

        # Remove assigned patients
        patient_ids = [pid for pid in patient_ids if pid not in fold_patients]

        val_data = data[data[patient_id_column].isin(fold_patients)]
        train_data = data[~data[patient_id_column].isin(fold_patients)]

        folds.append((train_data, val_data))

    return folds
