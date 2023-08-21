import os
import csv
import pandas as pd
import json

input_folder = '/input'
config_file = '/config/IDs.json'

with open(config_file) as f:
    config = json.load(f)

imaging_occurrence_id_start = config["imaging_occurrence_id_start"]
imaging_feature_id_start = config["imaging_feature_id_start"]
imaging_feature_domain_id_start = config["measurement_id_start"]

imaging_feature_id = -1
imaging_feature_domain_id = -1

patient_folders = [folder for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, folder))]

for patient_folder in patient_folders:
    patient_folder_path = os.path.join(input_folder, patient_folder)
    person_id = int(patient_folder)

    procedure_folders = [folder for folder in os.listdir(patient_folder_path) if os.path.isdir(os.path.join(patient_folder_path, folder))]

    for index, procedure_folder in enumerate(procedure_folders, start=1):
        procedure_folder_path = os.path.join(patient_folder_path, procedure_folder)
        
        if imaging_feature_id != -1:
            imaging_feature_id_start = imaging_feature_id + 1
        if imaging_feature_domain_id != -1:
            imaging_feature_domain_id_start = imaging_feature_domain_id + 1

        rows = []

        img_occ_file = os.path.join(procedure_folder_path, "omop_tables", "imaging_occurrence.csv")
        with open(img_occ_file, newline='') as img_occ_file:
            ids_reader = csv.DictReader(img_occ_file, delimiter=',')
 
            for row in ids_reader:
                imaging_occurrence_id = row['imaging_occurrence_id']

        for i, j in zip(range(imaging_feature_id_start, imaging_feature_id_start + 111), range(imaging_feature_domain_id_start, imaging_feature_domain_id_start + 111)):
            imaging_feature_id = i
            imaging_feature_domain_id = j

            row = {
                'imaging_feature_id': imaging_feature_id,
                'imaging_finding_num': 0,  #dx timepoint
                'imaging_occurrence_id': int(imaging_occurrence_id),
                'domain_concept_id': 21,  #measurement table code
                'imaging_feature_domain_id': imaging_feature_domain_id,
                'anatomic_site_location': '2000000026',  #change this value according to the vocabulary
                #TODO: define with IMPaCT group:
                'alg_system': 'https://github.com',  #example 
                'alg_datetime': '16-06-2023'  #example
            }

            rows.append(row)

        omop_tables_folder = os.path.join(procedure_folder_path, "omop_tables")
        os.makedirs(omop_tables_folder, exist_ok=True)

        csv_file = os.path.join(omop_tables_folder, "imaging_feature.csv")
        fieldnames = rows[0].keys()

        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"The file '{csv_file}' has been successfully created.")

        df = pd.DataFrame(rows)

        excel_file = os.path.join(omop_tables_folder, 'imaging_feature.xlsx')
        df.to_excel(excel_file, index=False)

        print(f"The file '{excel_file}' has been successfully created.")
