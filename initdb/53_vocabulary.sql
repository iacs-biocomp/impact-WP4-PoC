INSERT INTO omop.vocabulary (vocabulary_id,vocabulary_name,vocabulary_reference,vocabulary_version,vocabulary_concept_id) VALUES
	 ('ABMS','Provider Specialty (American Board of Medical Specialties)','http://www.abms.org/member-boards/specialty-subspecialty-certificates','2018-06-26 ABMS',45756746),
	 ('AMT','Australian Medicines Terminology (NEHTA)','https://www.nehta.gov.au/implementation-resources/terminology-access','Clinical Terminology v20210630',238),
	 ('APC','Ambulatory Payment Classification (CMS)','http://www.cms.gov/Medicare/Medicare-Fee-for-Service-Payment/HospitalOutpatientPPS/Hospital-Outpatient-Regulations-and-Notices.html','2018-January-Addendum-A',44819132),
	 ('ATC','WHO Anatomic Therapeutic Chemical Classification','http://www.whocc.no/atc_ddd_index/','RxNorm 20210907',44819117),
	 ('BDPM','Public Database of Medications (Social-Sante)','http://base-donnees-publique.medicaments.gouv.fr/telechargement.php','BDPM 20191006',236),
	 ('CCAM','Common Classification of Medical Acts','https://www.ameli.fr/accueil-de-la-ccam/telechargement/index.php','CCAM version 64',32957),
	 ('CDM','OMOP Common DataModel','https://github.com/OHDSI/CommonDataModel','CDM v6.0.0',32485),
	 ('CGI','Cancer Genome Interpreter (Pompeu Fabra University)','https://www.cancergenomeinterpreter.org/data/cgi_biomarkers_latest.zip','CGI20180216',32914),
	 ('CIEL','Columbia International eHealth Laboratory (Columbia University)','https://wiki.openmrs.org/display/docs/Getting+and+Using+the+MVP-CIEL+Concept+Dictionary','Openmrs 1.11.0 20150227',45905710),
	 ('CIM10','International Classification of Diseases, Tenth Revision, French Edition','https://www.atih.sante.fr/nomenclatures-de-recueil-de-linformation/cim','CIM10 2022',32806);
INSERT INTO omop.vocabulary (vocabulary_id,vocabulary_name,vocabulary_reference,vocabulary_version,vocabulary_concept_id) VALUES
	 ('CIViC','Clinical Interpretation of Variants in Cancer (civicdb.org)','https://github.com/griffithlab/civic-server/blob/master/public/downloads/RankedCivicGeneCandidates.tsv','CIViC 2022-10-01',32913),
	 ('CMS Place of Service','Place of Service Codes for Professional Claims (CMS)','http://www.cms.gov/Medicare/Medicare-Fee-for-Service-Payment/PhysicianFeeSched/downloads//Website_POS_database.pdf','2009-01-11',44819110),
	 ('CPT4','Current Procedural Terminology version 4 (AMA)','http://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html','2022 Release',44819100),
	 ('CTD','Comparative Toxicogenomic Database','http://ctdbase.org','CTD 2020-02-19',32735),
	 ('CVX','CDC Vaccine Administered CVX (NCIRD)','https://www2a.cdc.gov/vaccines/iis/iisstandards/vaccines.asp?rpt=cvx','CVX 20221019',581400),
	 ('Cancer Modifier','Diagnostic Modifiers of Cancer (OMOP)','OMOP generated','Cancer Modifier 20220909',32929),
	 ('ClinVar','ClinVar (NCBI)','https://ftp.ncbi.nlm.nih.gov/pub/clinvar/','ClinVar v20200901',32915),
	 ('Cohort','Legacy OMOP HOI or DOI cohort','OMOP generated',NULL,44819123),
	 ('Cohort Type','OMOP Cohort Type','OMOP generated',NULL,44819234),
	 ('Concept Class','OMOP Concept Class','OMOP generated',NULL,44819233);
INSERT INTO omop.vocabulary (vocabulary_id,vocabulary_name,vocabulary_reference,vocabulary_version,vocabulary_concept_id) VALUES
	 ('Condition Status','OMOP Condition Status','OMOP generated',NULL,32887),
	 ('Condition Type','OMOP Condition Occurrence Type','OMOP generated',NULL,44819127),
	 ('Cost','OMOP Cost','OMOP generated',NULL,581457),
	 ('Cost Type','OMOP Cost Type','OMOP generated',NULL,5029),
	 ('Currency','International Currency Symbol (ISO 4217)','http://www.iso.org/iso/home/standards/currency_codes.htm','2008',44819153),
	 ('DPD','Drug Product Database (Health Canada)','http://open.canada.ca/data/en/dataset/bf55e42a-63cb-4556-bfd8-44f26e5a36fe','DPD 25-JUN-17',231),
	 ('DRG','Diagnosis-related group (CMS)','http://www.cms.gov/Medicare/Medicare-Fee-for-Service-Payment/AcuteInpatientPPS/Acute-Inpatient-Files-for-Download.html','2011-18-02',44819130),
	 ('Death Type','OMOP Death Type','OMOP generated',NULL,44819135),
	 ('Device Type','OMOP Device Type','OMOP generated',NULL,44819151),
	 ('Domain','OMOP Domain','OMOP generated',NULL,44819147);
INSERT INTO omop.vocabulary (vocabulary_id,vocabulary_name,vocabulary_reference,vocabulary_version,vocabulary_concept_id) VALUES
	 ('Drug Type','OMOP Drug Exposure Type','OMOP generated',NULL,44819126),
	 ('EDI','Korean EDI','http://www.hira.or.kr/rd/insuadtcrtr/bbsView.do?pgmid=HIRAA030069000400&brdScnBltNo=4&brdBltNo=51354&pageIndex=1&isPopupYn=Y','EDI 2019.10.01',32736),
	 ('EphMRA ATC','Anatomical Classification of Pharmaceutical Products (EphMRA)','http://www.ephmra.org/Anatomical-Classification','EphMRA ATC 2016',243),
	 ('Episode','OMOP Episode','OMOP generated','Episode 20201014',32523),
	 ('Episode Type','OMOP Episode Type','OMOP generated',NULL,32542),
	 ('Ethnicity','OMOP Ethnicity','OMOP generated',NULL,44819134),
	 ('GCN_SEQNO','Clinical Formulation ID (FDB)','FDB US distribution package','20151119 Release',44819141),
	 ('GGR','Commented Drug Directory (BCFI)','http://www.bcfi.be/nl/download','GGR 20210901',581450),
	 ('Gender','OMOP Gender','OMOP generated',NULL,44819108),
	 ('HCPCS','Healthcare Common Procedure Coding System (CMS)','http://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html','20221001 Alpha Numeric HCPCS File',44819101);
INSERT INTO omop.vocabulary (vocabulary_id,vocabulary_name,vocabulary_reference,vocabulary_version,vocabulary_concept_id) VALUES
	 ('HES Specialty','Hospital Episode Statistics Specialty (NHS)','http://www.datadictionary.nhs.uk/data_dictionary/attributes/m/main_specialty_code_de.asp?shownav=0','2018-06-26 HES Specialty',44819145),
	 ('HemOnc','HemOnc','https://hemonc.org','HemOnc 2021-09-08',32613),
	 ('ICD10','International Classification of Diseases, Tenth Revision (WHO)','http://www.who.int/classifications/icd/icdonlineversions/en/','2021 Release',44819124),
	 ('ICD10CM','International Classification of Diseases, Tenth Revision, Clinical Modification (NCHS)','http://www.cdc.gov/nchs/icd/icd10cm.htm','ICD10CM FY2023 code descriptions',44819098),
	 ('ICD10CN','International Classification of Diseases, Tenth Revision, Chinese Edition','http://www.sac.gov.cn/was5/web/search?channelid=97779&templet=gjcxjg_detail.jsp&searchword=STANDARD_CODE=%27GB/T%2014396-2016%27','2016 Release',32740),
	 ('ICD10GM','International Classification of Diseases, Tenth Revision, German Edition','https://www.dimdi.de/dynamic/.downloads/klassifikationen/icd-10-gm','ICD10GM 2022',32928),
	 ('ICD10PCS','ICD-10 Procedure Coding System (CMS)','http://www.cms.gov/Medicare/Coding/ICD10/index.html','ICD10PCS 2021',44819125),
	 ('ICD9CM','International Classification of Diseases, Ninth Revision, Clinical Modification, Volume 1 and 2 (NCHS)','http://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/codes.html','ICD9CM v32 master descriptions',5046),
	 ('ICD9Proc','International Classification of Diseases, Ninth Revision, Clinical Modification, Volume 3 (NCHS)','http://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/codes.html','ICD9CM v32 master descriptions',44819099),
	 ('ICD9ProcCN','International Classification of Diseases, Ninth Revision, Chinese Edition, Procedures','http://chiss.org.cn/hism/wcmpub/hism1029/notice/201712/P020171225613285104950.pdf','2017 Release',32744);
INSERT INTO omop.vocabulary (vocabulary_id,vocabulary_name,vocabulary_reference,vocabulary_version,vocabulary_concept_id) VALUES
	 ('ICDO3','International Classification of Diseases for Oncology, Third Edition (WHO)','https://seer.cancer.gov/icd-o-3/','ICDO3 SEER Site/Histology Released 06/2020',581426),
	 ('JAX','The Clinical Knowledgebase (The Jackson Laboratory)','https://ckbhome.jax.org/','JAX v20200824',32916),
	 ('JMDC','Japan Medical Data Center Drug Code (JMDC)','OMOP generated','JMDC 2020-04-30',32557),
	 ('KCD7','Korean Classification of Diseases, 7th Revision','https://www.hira.or.kr/rd/insuadtcrtr/bbsView.do?pgmid=HIRAA030069000000&brdScnBltNo=4&brdBltNo=50760&pageIndex=1&isPopupYn=Y#none','7th revision',32688),
	 ('KDC','Korean Drug Code (HIRA)','https://www.hira.or.kr/eng/','KDC 2020-07-31',32422),
	 ('KNHIS','Korean National Health Information System','OMOP generated',NULL,32723),
	 ('Korean Revenue Code','Korean Revenue Code','OMOP generated',NULL,32724),
	 ('LOINC','Logical Observation Identifiers Names and Codes (Regenstrief Institute)','http://loinc.org/downloads/loinc','LOINC 2.72',44819102),
	 ('Language','Language (OMOP)','OMOP generated','Language 20221030',33069),
	 ('MDC','Major Diagnostic Categories (CMS)','http://www.cms.gov/Medicare/Medicare-Fee-for-Service-Payment/AcuteInpatientPPS/Acute-Inpatient-Files-for-Download.html','2013-01-06',44819131);
INSERT INTO omop.vocabulary (vocabulary_id,vocabulary_name,vocabulary_reference,vocabulary_version,vocabulary_concept_id) VALUES
	 ('MMI','Modernizing Medicine (MMI)','MMI proprietary',NULL,581367),
	 ('MeSH','Medical Subject Headings (NLM)','http://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html','2022 Release',44819136),
	 ('Meas Type','OMOP Measurement Type','OMOP generated',NULL,44819152),
	 ('Medicare Specialty','Medicare provider/supplier specialty codes (CMS)','http://www.cms.gov/Medicare/Provider-Enrollment-and-Certification/MedicareProviderSupEnroll/Taxonomy.html','2018-06-26 Specialty',44819138),
	 ('Metadata','Metadata','OMOP generated',NULL,32675),
	 ('Multum','Cerner Multum (Cerner)','http://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html','2013-07-10',44819112),
	 ('NAACCR','Data Standards & Data Dictionary Volume II (NAACCR)','http://datadictionary.naaccr.org/?c=10','NAACCR v18',32642),
	 ('NCCD','Normalized Chinese Clinical Drug','https://www.ohdsi.org/wp-content/uploads/2020/07/NCCD_RxNorm_Mapping_0728.pdf','NCCD_v02_2020',32807),
	 ('NCIt','NCI Thesaurus (National Cancer Institute)','http://evs.nci.nih.gov/ftp1/NCI_Thesaurus',' NCIt 20220509',32917),
	 ('NDC','National Drug Code (FDA and manufacturers)','http://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html, http://www.fda.gov/downloads/Drugs/DevelopmentApprovalProcess/UCM070838.zip','NDC 20221030',44819105);
INSERT INTO omop.vocabulary (vocabulary_id,vocabulary_name,vocabulary_reference,vocabulary_version,vocabulary_concept_id) VALUES
	 ('NDFRT','National Drug File - Reference Terminology (VA)','http://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html','RXNORM 2018-08-12',44819103),
	 ('NFC','New Form Code (EphMRA)','http://www.ephmra.org/New-Form-Codes-Classification','NFC 20160704',245),
	 ('NUCC','National Uniform Claim Committee Health Care Provider Taxonomy Code Set (NUCC)','http://www.nucc.org/index.php?option=com_content&view=article&id=107&Itemid=132','2018-06-26 NUCC',44819137),
	 ('Nebraska Lexicon','Nebraska Lexicon','https://www.unmc.edu/pathology-research/bioinformatics/campbell/tdc.html','Nebraska Lexicon 20190816',32757),
	 ('None','OMOP Standardized Vocabularies','OMOP generated','v5.0 31-OCT-22',44819096),
	 ('Note Type','OMOP Note Type','OMOP generated',NULL,44819146),
	 ('OMOP Extension','OMOP Extension (OHDSI)','OMOP generated','OMOP Extension 20221026',32758),
	 ('OMOP Genomic','OMOP Genomic vocabulary','OMOP generated','OMOP Genomic 20210727',33002),
	 ('OMOP Invest Drug','OMOP Investigational Drugs','https://gsrs.ncats.nih.gov/, https://gsrs.ncats.nih.gov/#/release','OMOP Invest Drug version 2022-05-12',33051),
	 ('OPCS4','OPCS Classification of Interventions and Procedures version 4 (NHS)','http://systems.hscic.gov.uk/data/clinicalcoding/codingstandards/opcs4','2021 Release',44819143);
INSERT INTO omop.vocabulary (vocabulary_id,vocabulary_name,vocabulary_reference,vocabulary_version,vocabulary_concept_id) VALUES
	 ('OPS','Operations and Procedures Classification (OPS)','https://www.dimdi.de/dynamic/de/klassifikationen/ops/','OPS Version 2020',32956),
	 ('OSM','OpenStreetMap','https://www.openstreetmap.org/copyright/en, https://wambachers-osm.website/boundaries/','OSM Release 2019-02-21',32541),
	 ('OXMIS','Oxford Medical Information System (OCHP)','Codes extracted from GPRD database, courtesy of GSK',NULL,44819114),
	 ('Obs Period Type','OMOP Observation Period Type','OMOP generated',NULL,44819149),
	 ('Observation Type','OMOP Observation Type','OMOP generated',NULL,44819129),
	 ('OncoKB','Oncology Knowledge Base (MSK)','https://www.oncokb.org/','OncoKB v20210502',32999),
	 ('OncoTree','OncoTree (MSK)','http://oncotree.mskcc.org/','OncoTree version 2021-11-02',33008),
	 ('PCORNet','National Patient-Centered Clinical Research Network (PCORI)','OMOP generated',NULL,44819148),
	 ('PPI','AllOfUs_PPI (Columbia)','http://terminology.pmi-ops.org/CodeSystem/ppi','Codebook Version 0.4.43 + COVID + MHWB + SDOH + PFH',581404),
	 ('Plan','Health Plan - contract to administer healthcare transactions by the payer, facilitated by the sponsor','OMOP generated',NULL,32471);
INSERT INTO omop.vocabulary (vocabulary_id,vocabulary_name,vocabulary_reference,vocabulary_version,vocabulary_concept_id) VALUES
	 ('Plan Stop Reason','Plan Stop Reason - Reason for termination of the Health Plan','OMOP generated',NULL,32474),
	 ('Procedure Type','OMOP Procedure Occurrence Type','OMOP generated',NULL,44819128),
	 ('Provider','OMOP Provider','OMOP generated',NULL,32573),
	 ('Race','Race and Ethnicity Code Set (USBC)','http://www.cdc.gov/nchs/data/dvs/Race_Ethnicity_CodeSet.pdf','Version 1.0',44819109),
	 ('Read','NHS UK Read Codes Version 2 (HSCIC)','http://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html','NHS READV2 21.0.0 20160401000001 + DATAMIGRATION_25.0.0_20180403000001',44819113),
	 ('Relationship','OMOP Relationship','OMOP generated',NULL,44819235),
	 ('Revenue Code','UB04/CMS1450 Revenue Codes (CMS)','http://www.mpca.net/?page=ERC_finance','2010 Release',44819133),
	 ('RxNorm','RxNorm (NLM)','http://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html','RxNorm 20221003',44819104),
	 ('RxNorm Extension','RxNorm Extension (OHDSI)','OMOP generated','RxNorm Extension 2022-10-30',252),
	 ('SMQ','Standardised MedDRA Queries (MSSO)','http://www.meddramsso.com/secure/smq/SMQ_Spreadsheet_14_0_English_update.xlsx','Version 14.0',44819121);
INSERT INTO omop.vocabulary (vocabulary_id,vocabulary_name,vocabulary_reference,vocabulary_version,vocabulary_concept_id) VALUES
	 ('SNOMED','Systematic Nomenclature of Medicine - Clinical Terms (IHTSDO)','http://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html','2021-07-31 SNOMED CT International Edition; 2021-09-01 SNOMED CT US Edition; 2021-11-24 SNOMED CT UK Edition',44819097),
	 ('SNOMED Veterinary','SNOMED Veterinary','https://vtsl.vetmed.vt.edu/extension/','SNOMED Veterinary 20190401',32549),
	 ('SOPT','Source of Payment Typology (PHDSC)','https://www.nahdo.org/sopt','SOPT Version 9.2',32473),
	 ('SPL','Structured Product Labeling (FDA)','http://www.fda.gov/Drugs/InformationOnDrugs/ucm142438.htm','NDC 20221030',44819140),
	 ('Specimen Type','OMOP Specimen Type','OMOP generated',NULL,581376),
	 ('Sponsor','Sponsor - institution or individual financing healthcare transactions','OMOP generated',NULL,32472),
	 ('Supplier','OMOP Supplier','OMOP generated',NULL,32574),
	 ('Type Concept','OMOP Type Concept','OMOP generated','Type Concept 20221030',32808),
	 ('UB04 Point of Origin','UB04 Claim Source Inpatient Admission Code (CMS)','https://www.resdac.org/cms-data/variables/Claim-Source-Inpatient-Admission-Code',NULL,32045),
	 ('UB04 Pri Typ of Adm','UB04 Claim Inpatient Admission Type Code (CMS)','https://www.resdac.org/cms-data/variables/Claim-Inpatient-Admission-Type-Code',NULL,32046);
INSERT INTO omop.vocabulary (vocabulary_id,vocabulary_name,vocabulary_reference,vocabulary_version,vocabulary_concept_id) VALUES
	 ('UB04 Pt dis status','UB04 Patient Discharge Status Code (CMS)','https://www.resdac.org/cms-data/variables/patient-discharge-status-code',NULL,32047),
	 ('UB04 Typ bill','UB04 Type of Bill - Institutional (USHIK)','https://ushik.ahrq.gov/ViewItemDetails?&system=apcd&itemKey=196987000',NULL,32044),
	 ('UCUM','Unified Code for Units of Measure (Regenstrief Institute)','http://aurora.regenstrief.org/~ucum/ucum.html#section-Alphabetic-Index','Version 1.8.2',44819107),
	 ('UK Biobank','UK Biobank','https://biobank.ctsu.ox.ac.uk/showcase/schema.cgi; https://biobank.ctsu.ox.ac.uk/crystal/refer.cgi?id=141140','version 2020-10-15',32976),
	 ('US Census','United States Census Bureau','https://www.census.gov/geo/maps-data/data/tiger-cart-boundary.html','US Census 2017 Release',32570),
	 ('VA Class','VA National Drug File Class (VA)','http://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html','RxNorm 20211101',44819122),
	 ('VANDF','Veterans Health Administration National Drug File','http://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html','RxNorm 20211004',44819120),
	 ('Visit','OMOP Visit','OMOP generated','Visit 20211216',44819119),
	 ('Visit Type','OMOP Visit Type','OMOP generated',NULL,44819150),
	 ('Vocabulary','OMOP Vocabulary','OMOP generated',NULL,44819232);
INSERT INTO omop.vocabulary (vocabulary_id,vocabulary_name,vocabulary_reference,vocabulary_version,vocabulary_concept_id) VALUES
	 ('dm+d','Dictionary of Medicines and Devices (NHS)','https://isd.hscic.gov.uk/trud3/user/authenticated/group/0/pack/1/subpack/24/releases','dm+d Version 6.3.0 20210628',232),
	 ('RADLEX','Radiology Lexicon','https://radlex.org/','RADLEX Version 4.1',2000000000),
	 ('IMPACT','IMPaCT-Data vocabulary','','IMPACT Data Version 1.0',2000000100),
	 ('SNOMED-SARS-CoV2','SNOMED-Extension for SARS-CoV2','https://www.sanidad.gob.es/areas/saludDigital/interoperabilidadSemantica/docs/Conceptos_relacionados_SARS-CoV-2-Version16.0_20220601_es.pdf','SNOMED CT-SARS-CoV-2-Version16',2000000500);







