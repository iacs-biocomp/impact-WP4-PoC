set datestyle = euro;
copy omop.imaging_occurrence from '/data/imagen/imaging_occurrence.csv' with delimiter ',' csv header;
copy omop.imaging_feature from '/data/imagen/imaging_feature.csv' with delimiter ',' csv header;
copy omop.measurement from '/data/imagen/measurement.csv' with delimiter ',' csv header;