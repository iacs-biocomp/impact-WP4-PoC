import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
import json
import radiomics
import nrrd
import pydicom
import datetime

from radiomics import featureextractor

params = {}
patients_path = "/input"
exteractor = featureextractor.RadiomicsFeatureExtractor(geometryTolerance=1.0)

def extraer_tag(dcm, tag):
    # Check if the tag exists
    compare_tag = " ".join(map(str, tag))
    tag_date="8 33"
    if compare_tag == tag_date:  
        try:
            outcome = dcm.get(tag).value
        except:
            outcome = datetime.date.today()
    else:
        try:
            outcome = dcm.get(tag).value
        except:
            outcome = 0
    return outcome

list_patients = os.listdir(patients_path)
for patient in list_patients:
    list_procedures = os.listdir(os.path.join(patients_path, patient))
    for procedure in list_procedures:
        procedure_path = os.path.join(patients_path, patient, procedure)
        images_path = os.path.join(procedure_path, "original")
        masks_path = os.path.join(procedure_path, "segmentation")
        radiomics_path = os.path.join(procedure_path, "radiomics")
        dicom_path = os.path.join(procedure_path, "dicom_headers")
        if not os.path.exists(dicom_path):
            os.mkdir(dicom_path)
        if not os.path.exists(radiomics_path):
            os.mkdir(radiomics_path)
        # Extract the following dicom tags from the first dicom file of the procedure: (0008,0021),(0008,0018),(0018,5101),(0018,1050),(0028,0011),(0028,0010)
        # añadir el primer archivo dicom que encuentre al path_image
        path_image = os.path.join(images_path, os.listdir(images_path)[0])
        dicom = pydicom.dcmread(path_image)
        dicom_tags = {}
        dicom_tags["imaging_occurrence_date"] = extraer_tag(dicom, (0x0008, 0x0021))
        dicom_tags["imaging_study_uid"] = extraer_tag(dicom, (0x0008, 0x0018))
        dicom_tags["view_position"] = extraer_tag(dicom, (0x0018, 0x5101))
        dicom_tags["spatial_resolution"] = extraer_tag(dicom, (0x0018, 0x1050))
        dicom_tags["columns"] = extraer_tag(dicom, (0x0028, 0x0011))
        dicom_tags["rows"] = extraer_tag(dicom, (0x0028, 0x0010))
        #save the dicom tags in a csv file
        df = pd.DataFrame.from_dict(dicom_tags, orient="index")
        df = df.transpose()
        df.to_csv(os.path.join(dicom_path, "dicom_tags.csv"), index=False)

        path_image = os.path.join(images_path, os.listdir(images_path)[0])
        path_mask = os.path.join(masks_path, os.listdir(masks_path)[0])
        image = sitk.ReadImage(path_image)
        mask = sitk.ReadImage(path_mask)
        image_size = image.GetSize()
        # add a 3rd dimension to the mask with value 1
        mask = sitk.JoinSeries(mask)
        result = exteractor.execute(image, mask)
        # Save the results that starts with "original_" in a dataframe
        df = pd.DataFrame.from_dict({k: v for k, v in result.items() if k.startswith("original_")}, orient="index")
        # Transpose the dataframe to have the features as columns
        df = df.transpose()
        # Delete the original_ prefix from the column names
        df.columns = df.columns.str.replace("original_", "")
        # Save the dataframe as a csv file
        df.to_csv(os.path.join(radiomics_path, "radiomics.csv"), index=False)
        print(patient + "." + procedure + "csv de radiómicas generado")