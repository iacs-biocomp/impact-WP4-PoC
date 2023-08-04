--IMAGING_OCCURRENCE TABLE
CREATE TABLE IMAGING_OCCURRENCE (
			imaging_occurrence_id integer NOT NULL,
			person_id integer NOT NULL,
			procedure_occurrence_id integer NOT NULL,
			wadors_uri varchar(255) NULL,
			imaging_occurrence_date date NOT NULL,
			imaging_study_UID varchar(255) NULL,
			imaging_series_UID varchar(255) NULL,
			modality varchar(255) NOT NULL, 
			anatomic_site_concept_id varchar(255) NULL 
			);

--IMAGING_FEATURE TABLE
CREATE TABLE IMAGING_FEATURE (
			imaging_feature_id integer NOT NULL,
			imaging_finding_num integer NOT NULL,
			imaging_occurrence_id integer NULL,
			domain_concept_id integer NOT NULL,
			imaging_feature_domain_id varchar(255) NOT NULL,
			anatomic_site_concept_id varchar(255) NULL,
			alg_system varchar(255) NULL,
			alg_datetime date NOT NULL
			);


--PRIMARY KEYS
ALTER TABLE IMAGING_OCCURRENCE ADD CONSTRAINT xpk_IMAGING_OCCURRENCE PRIMARY KEY (imaging_occurrence_id);

ALTER TABLE IMAGING_FEATURE ADD CONSTRAINT xpk_IMAGING_FEATURE PRIMARY KEY (imaging_feature_id);

-- FOREIGN KEYS
ALTER TABLE IMAGING_OCCURRENCE ADD CONSTRAINT fpk_IMAGING_OCCURRENCE_person_id FOREIGN KEY (person_id) REFERENCES PERSON(person_id) ON DELETE CASCADE;

ALTER TABLE IMAGING_OCCURRENCE ADD CONSTRAINT fpk_IMAGING_OCCURRENCE_procedure_occurrence_id FOREIGN KEY (procedure_occurrence_id) REFERENCES PROCEDURE_OCCURRENCE(procedure_occurrence_id) ON DELETE CASCADE;

ALTER TABLE IMAGING_FEATURE ADD CONSTRAINT fpk_IMAGING_FEATURE_IMAGING_OCCURRENCE_id FOREIGN KEY (imaging_occurrence_id) REFERENCES IMAGING_OCCURRENCE(imaging_occurrence_id) ON DELETE CASCADE;
