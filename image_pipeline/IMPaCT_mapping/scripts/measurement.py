import csv
import os
import pandas as pd
import numpy as np
import pytz
import json
from datetime import datetime, timedelta
from map_radiomics import map_radiomics
from map_dicom_headers import map_dicom_headers

input_folder = '/input'

patient_folders = [folder for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, folder))]

rows = []

for patient_folder in patient_folders:
    patient_folder_path = os.path.join(input_folder, patient_folder)
    person_id = int(patient_folder)

    procedure_folders = [folder for folder in os.listdir(patient_folder_path) if os.path.isdir(os.path.join(patient_folder_path, folder))]

    for index, procedure_folder in enumerate(procedure_folders, start=1):
        procedure_folder_path = os.path.join(patient_folder_path, procedure_folder)
        
        now = datetime.now() + timedelta(hours=2)
        local_tz = pytz.timezone('Europe/Madrid')
        measurement_datetime = local_tz.localize(now)
        measurement_datetime = measurement_datetime.astimezone(pytz.utc)
        measurement_date = measurement_datetime.date()
        measurement_datetime_str = measurement_datetime.strftime("%Y-%m-%d %H:%M:%S")
        measurement_time_str = measurement_datetime.strftime("%H:%M:%S")

        measurement_ids=[]
        img_feature_file = os.path.join(procedure_folder_path, "omop_tables", "imaging_feature.csv")
        with open(img_feature_file, newline='') as img_feature:
            measurement = csv.DictReader(img_feature, delimiter=',')
            for id in measurement:
                measurement_id = id.get('imaging_feature_domain_id')
                measurement_ids.append(int(measurement_id)) 

        rad_input_file = os.path.join(procedure_folder_path, "radiomics", "radiomics.csv")
        rad_output_file = os.path.join(procedure_folder_path, "radiomics", "radiomics_mapped.csv")
        map_radiomics(rad_input_file, rad_output_file)
        
        dcm_headers_input_file = os.path.join(procedure_folder_path, "dicom_headers", "dicom_tags.csv")
        dcm_headers_output_file = os.path.join(procedure_folder_path, "dicom_headers", "dicom_tags_mapped.csv")
        map_dicom_headers(dcm_headers_input_file, dcm_headers_output_file)

        df_rad = pd.read_csv(rad_output_file)
        df_dcm_headers = pd.read_csv(dcm_headers_output_file)
        df_features = pd.concat([df_rad, df_dcm_headers], ignore_index=True)
        df_features = df_features.replace({np.nan:None})

        for i, row in df_features.iterrows():
            measurement_id = measurement_ids[i]
            measurement_source_value = row['feature']
            measurement_concept_id = int(row['concept_id'])
            measurement_source_concept_id = row['source_concept_id']
            value_as_number = row['value_as_number']
            value_as_concept = row['value_as_concept']
            measurement_type_concept_id = row['measurement_type_concept_id']

            row_data = {
                'measurement_id': int(measurement_id),
                'person_id': int(person_id),
                'measurement_concept_id': int(measurement_concept_id),
                'measurement_date': measurement_date,
                'measurement_datetime': measurement_datetime_str,
                'measurement_time': measurement_time_str,
                'measurement_type_concept_id': measurement_type_concept_id, 
                'operator_concept_id': int('4172703'),  #'=' code  276136004 (SNOMED), 4172703 (OMOP concept_id)
                'value_as_number': value_as_number,
                'value_as_concept': value_as_concept,
                'unit_concept_id': '',
                'range_low': '',
                'range_high': '',
                'provider_id': '',
                'visit_occurrence_id': '',
                'visit_detail_id': '',
                'measurement_source_value': measurement_source_value,
                'measurement_source_concept_id': 0, # habría que obtener el concept_id para cada source code 
                'unit_source_value': '',
                #'unit_source_concept_id': '',
                'value_source_value': '',
                #'measurement_event_id': '',
                #'meas_event_field_concept_id': ''
            }
            rows.append(row_data)

        omop_tables_folder = os.path.join(procedure_folder_path, "omop_tables")
        os.makedirs(omop_tables_folder, exist_ok=True)

        csv_file = os.path.join(omop_tables_folder, "measurement.csv")
        fieldnames = rows[0].keys()
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"The file '{csv_file}' has been successfully created.")

        df = pd.DataFrame(rows)

        excel_file = os.path.join(omop_tables_folder, 'measurement.xlsx')
        df.to_excel(excel_file, index=False)

        print(f"The file '{excel_file}' has been successfully created.")
        rows.clear()