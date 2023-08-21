import os
import csv
import pandas as pd
import json

input_folder = '/input'
config_file = '/config/IDs.json'

with open(config_file) as f:
    config = json.load(f)

imaging_occurrence_id_start = config["imaging_occurrence_id_start"]
imaging_occurrence_id = -1

patient_folders = [folder for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, folder))]

for patient_folder in patient_folders:
    patient_folder_path = os.path.join(input_folder, patient_folder)
    person_id = int(patient_folder)

    procedure_folders = [folder for folder in os.listdir(patient_folder_path) if os.path.isdir(os.path.join(patient_folder_path, folder))]
    if imaging_occurrence_id != -1:
        imaging_occurrence_id_start = imaging_occurrence_id + 1
    
    for index, procedure_folder in enumerate(procedure_folders, start=1):
        imaging_occurrence_id = imaging_occurrence_id_start + index - 1
        procedure_folder_path = os.path.join(patient_folder_path, procedure_folder)
        procedure_occurrence_id = int(procedure_folder)

        wadors_uri = os.path.join(procedure_folder_path, "original")

        dicom_headers_file = os.path.join(procedure_folder_path, "dicom_headers", "dicom_tags.csv")
        with open(dicom_headers_file, newline='') as dicom_file:
            dicom_reader = csv.DictReader(dicom_file, delimiter=',')
            dicom_headers = next(dicom_reader)

            imaging_occurrence_date = dicom_headers.get('imaging_occurrence_date', '')
            imaging_study_UID = dicom_headers.get('imaging_study_uid', '')
            imaging_series_UID = imaging_study_UID #ad-hoc for rx images

        rows = []

        row = {
            'imaging_occurrence_id': imaging_occurrence_id,
            'person_id': person_id,
            'procedure_occurrence_id': procedure_occurrence_id,
            'wadors_uri': wadors_uri,
            'imaging_occurrence_date': str(imaging_occurrence_date),
            'imaging_study_UID': str(imaging_study_UID),
            'imaging_series_UID': str(imaging_series_UID),
            'modality': 'RX',
            'anatomic_site_location': '2000000026'  #change this value according to the vocabulary
        }

        rows.append(row)

        omop_tables_folder = os.path.join(procedure_folder_path, "omop_tables")
        os.makedirs(omop_tables_folder, exist_ok=True)

        csv_file = os.path.join(omop_tables_folder, "imaging_occurrence.csv")
        fieldnames = rows[0].keys()

        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"The file '{csv_file}' has been successfully created.")

        df = pd.DataFrame(rows)

        excel_file = os.path.join(omop_tables_folder,'imaging_occurrence.xlsx')
        df.to_excel(excel_file, index=False)

        print(f"The file '{excel_file}' has been successfully created.")

        df = pd.DataFrame(rows)

        omop_tables_folder = os.path.join(procedure_folder_path, "omop_tables")
        os.makedirs(omop_tables_folder, exist_ok=True)
        csv_file = os.path.join(omop_tables_folder, "imaging_occurrence.csv")

        df.to_csv(csv_file, index=False)