import pandas as pd

def map_dicom_headers(input_file, output_file):
    df_original = pd.read_csv(input_file)

    df_nuevo = pd.DataFrame(columns=['feature', 'value_as_number','value_as_concept', 'concept_id', 'source_concept_id','measurement_type_concept_id'])

    concept_ids = []
    source_concept_ids = []
    #Manually defined concept_ids and source_concept_ids #TODO: automatization
    concept_ids = [2000000301, 2000000302, 2000000303, 2000000304]
    source_concept_ids = ['(0018, 5101)', '(0018, 1050)', '(0028, 0011)', '(0028, 0010)']

    columns_to_map = df_original.columns[2:]

    if len(columns_to_map) <= len(concept_ids) and len(columns_to_map) <= len(source_concept_ids):
        for index, row in df_original.iterrows():
            for i, column in enumerate(columns_to_map):
                feature = column
                value = row[column]
                concept_id = concept_ids[i]
                source_concept_id = source_concept_ids[i]

                if value == 'PA' or value == 'AP':
                    value_as_concept = 2000000305 #change this value according to the vocabulary
                    value_as_number = None
                else:
                    value_as_concept = None
                    value_as_number = float(value)

                new_row = pd.DataFrame({
                    'feature': [feature],
                    'value_as_number': [value_as_number],
                    'value_as_concept': [value_as_concept],
                    'concept_id': [concept_id],
                    'source_concept_id': [source_concept_id],
                    'measurement_type_concept_id':2000000300 #concept_class DICOM code
                })
                df_nuevo = pd.concat([df_nuevo, new_row], ignore_index=True)

        df_nuevo.to_csv(output_file, index=False)
    else:
        print("There are not enough concept_ids or source_concept_ids to map all columns.")