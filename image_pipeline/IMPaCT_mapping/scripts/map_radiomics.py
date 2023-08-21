import pandas as pd

def map_radiomics(input_file, output_file):
    df_original = pd.read_csv(input_file)

    df_new = pd.DataFrame(columns=['feature', 'value_as_number', 'value_as_concept', 'concept_id', 'source_concept_id', 'measurement_type_concept_id'])

    #Define ranges for concept_id and source_concept_id #TODO: change according to the created vocabulary
    concept_id_start = 2000000102
    concept_id_end = 2000000208
    source_concept_id_start = 'RAD001'
    source_concept_id_end = 'RAD107'

    current_concept_id = concept_id_start
    current_source_concept_id = source_concept_id_start

    def get_next_source_concept_id(current_source_concept_id):
        prefix = current_source_concept_id[:-3]
        number = int(current_source_concept_id[-3:])
        next_number = number + 1
        return prefix + str(next_number).zfill(3)

    for index, row in df_original.iterrows():
        for column in df_original.columns:
            feature = column
            value = row[column]

            new_row = pd.DataFrame({
                'feature': [feature],
                'value_as_number': [value],
                'value_as_concept': None,
                'concept_id': [current_concept_id],
                'source_concept_id': [current_source_concept_id],
                'measurement_type_concept_id': 2000000101 #Concept_class Radiomics code
            })
            df_new = pd.concat([df_new, new_row], ignore_index=True)

            current_concept_id += 1
            current_source_concept_id = get_next_source_concept_id(current_source_concept_id)

    df_new.to_csv(output_file, index=False)
