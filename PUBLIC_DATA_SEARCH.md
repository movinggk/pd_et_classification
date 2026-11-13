# Public Parkinson's disease and essential tremor datasets: systematic search

**Checked: 6 October 2026.** PD = Parkinson's disease; ET = essential tremor; HC = healthy controls; DBS = deep brain stimulation.

**PADS acquisition update, 7 October 2026:** the complete v1.0.0 release has now been downloaded to [`data/pads/1.0.0/`](data/pads/1.0.0/), with all 12,689 officially checksummed files verified. The [acquisition guide](data/pads/README.md) documents the data, verification scope, and 304-participant PD/ET index. The search results and access checks below retain their original 6 October date.

## Scope and result

The search concerns **PD versus ET classification**, with ET-only datasets listed as supplementary resources. It is not an exhaustive catalog of the much larger PD-versus-HC literature.

Six sources were confirmed to release participant-level PD/ET recordings or measurements without an access application: PADS, Cologne accelerometry, Arm Tremor Database, caudate neurotransmitter data, digital spiral measurements, and cortical-plasticity/DTI measurements. The first three release movement-sensor time series. The other three use different modalities or derived measurements. B(RAIN)² supplies another mixed PD/ET resource with registration and a data-use agreement (DUA). Additional treatment-data and video/example resources have different evidential limits and are listed separately.

Here, **anonymous public download** means that a repository exposes data files without a researcher-access application or account requirement. This does not mean that reuse is unrestricted. **Registered/DUA access**, **author-request access**, **public metadata only**, and **unverified current downloads** are separate statuses.

During the original search, repository manifests, small clinical/annotation tables, licenses, and selected supplementary spreadsheets were inspected. Full raw signal, MRI, video, or genomic archives were not downloaded or validated at that stage. A download button or published file manifest was sufficient to identify a release, but an actual file inspection is stated separately below. The subsequent PADS download is documented in the acquisition update above.

## 1. Confirmed anonymous public sources containing both PD and ET

| Source | Independent participants | Released material | Access/license | Practical use and limits |
| --- | --- | --- | --- | --- |
| [PADS, PhysioNet v1.0.0](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/) | **276 PD + 28 ET**, within 469 overall | Both wrists: acceleration and gyroscope at 100 Hz during 11 active tasks; clinical/demographic metadata and questionnaire | Anonymous download; **CC BY-NC-SA 4.0** | Strongest starting point among the verified open wearable cohorts. Select the named diagnosis, not the combined differential-diagnosis numeric label. |
| [Cologne accelerometry, OSF](https://osf.io/mp5ha/) | **25 PD + 16 ET** according to the study; 54 limbs and repeated trials | Wrist acceleration at 100 Hz and finger acceleration at 2500 Hz; resting/postural tremor; CSV signals and annotations | Anonymous downloads; actual [license file](https://osf.io/download/z9t6d/) is **CC BY-NC-SA 4.0** | Additional PD/ET cohort, all DBS treated. Both hands and repeated trials must stay with the same participant in evaluation. |
| [Arm Tremor Database, Zenodo v2](https://zenodo.org/records/4772130) | **8 PD + 2 ET + 27 without a tremor diagnosis** | Four triaxial accelerometers at 62.5 Hz, placed at index finger, thumb, metacarpal region and wrist; four conditions; MATLAB structs and medication metadata | Anonymous download; **CC BY 4.0**; MATLAB file 6,749,196 bytes | Useful for prototyping. Two independent ET participants are insufficient for a convincing stand-alone diagnostic benchmark. |
| [Caudate neurotransmitter study, GitHub](https://github.com/AlecHartle/article-UG-Caudate) | **12 PD + 6 ET people**; **19 sessions** because one PD participant was recorded twice | Human intraoperative neurotransmitter estimates and social-exchange task data; MATLAB time-series file, trial CSV and clinical CSV, processing code | Public files; **no explicit repository data license found** | Genuine human PD/ET data, but invasive neurochemical measurements rather than wrist motion. Public accessibility alone does not establish broad reuse permission. |
| [Lin et al. digital spiral assessment, Figshare supplement](https://doi.org/10.6084/m9.figshare.5969692) | **12 PD + 12 ET labeled training participants**; 6 additional validation IDs marked `UN` | Per-drawing radial-change measures, velocity-spectrum area, physician ratings; subject IDs, hand, task, clinical scores; XLSX, 292,944 bytes | Public file downloaded and inspected; repository license label **“CC BY + CC0”** | Suitable for a small feature-based PD/ET experiment. The release inspected contains derived measurements, not raw x/y/time trajectories or a labeled image archive. The six `UN` validation IDs must not be assigned diagnoses by assumption. |
| [Lu et al. cortical plasticity and DTI, Figshare](https://doi.org/10.6084/m9.figshare.3812025) | **12 PD + 12 ET + 12 HC**, confirmed from the spreadsheet's subject IDs | Clinical data; motor-evoked-potential amplitude and intracortical-inhibition measurements before/after paired associative stimulation; corticospinal-tract fractional anisotropy and mean diffusivity; XLS, 106,496 bytes | Public file downloaded and inspected; **CC BY 4.0** | Individual measurements support exploratory tabular classification. This is not a release of raw EMG waveforms or MRI volumes. |

### Download links and verification details

**PADS.** The official [participant table](https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/preprocessed/file_list.csv) was downloaded and counted. Conditions: 276 `Parkinson's`, 28 `Essential Tremor`, 79 `Healthy`, 15 `Atypical Parkinsonism`, 11 `Multiple Sclerosis`, 60 `Other Movement Disorders`. `label == 2` includes multiple diagnoses; it is not an ET label. Some PD records mention ET in `disease_comment`, which requires a predefined diagnostic-review policy if overlap matters. The original dataset authors advise removing approximately the first 0.5 seconds of raw recordings because of watch vibration. [Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10770131/).

**Cologne.** [Training postural annotations](https://osf.io/download/46b3p/), [training resting annotations](https://osf.io/download/kg9xm/), [validation postural annotations](https://osf.io/download/yq275/), [validation resting annotations](https://osf.io/download/u5r6s/). The inspected training postural table has 478 repeated-trial rows: 244 PD and 234 ET. These are not participant counts. Some recording links are missing. The [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10611060/) reports CC BY-SA, but the downloadable repository license specifies the more restrictive CC BY-NC-SA. This catalog follows the actual license file.

**Arm Tremor.** [MATLAB data](https://zenodo.org/records/4772130/files/database.mat?download=1) and [README](https://zenodo.org/records/4772130/files/readme.txt?download=1). README entries 1–2 are ET, 3–10 are PD, 11–37 have no diagnosed tremor. The four conditions are rest, posture, free motion, and motion with an object. The README and metadata were inspected; the full MATLAB signal file was not loaded. [Study](https://doi.org/10.3390/signals2020016).

**Caudate neurotransmitters.** [Primary study](https://www.nature.com/articles/s41467-025-63079-w). Public repository paths: `Data/Behavioral_and_Neural/UG_Caudate_Raw_Data.mat`, `UG_Caudate_Table_R.csv`, `UG_Caudate_Clinical_Measures_R.csv`. The CSVs were downloaded and inspected. There are 570 trial rows and 19 recording IDs, with 30 trials per ID. The clinical table has 13 PD-labeled recording IDs and 6 ET-labeled IDs. The paper explicitly states 12 PD people and one repeat session; therefore, 13 PD recording IDs must not be reported as 13 independent PD people. The repeated-person mapping needs to be resolved before a participant-independent split. The article's CC BY-NC-ND license must not be assumed to license separately hosted repository data.

**Digital spirals.** [Workbook download](https://ndownloader.figshare.com/files/10703656); [primary study](https://link.springer.com/article/10.1186/s12883-018-1027-2). Three tasks: tracing a given spiral, freehand drawing, tracing a moving guide. The original acquisition used a Wacom tablet at 50 Hz; that is the source acquisition frequency, not evidence that the workbook contains 50-Hz raw samples. Seven sheets include `1)Testing Group`, `4)Validation Group`, `6)FTM score`, and `7) velocity`. Subject IDs establish 12 PD, 12 ET, and 6 `UN`. The license is reported exactly as the repository labels it; see the article's data-license paragraph for the CC0 waiver applied to data unless otherwise stated.

**Cortical plasticity/DTI.** [Spreadsheet download](https://ndownloader.figshare.com/files/5936232); [primary study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0162265). Sheets: `Basic Data`, `TMS`, `DTI`. `Basic Data` and `TMS` contain IDs `N-1`–`N-12`, `PD-1`–`PD-12`, `ET-1`–`ET-12`. Aggregate mean/SD rows are also present and must be excluded when constructing participant data. ET participants had intention tremor; medications were withheld under the study protocol. The paper calls the appended numbers raw data of the measurements, but they are subject-level measurements rather than original waveform/image files.

## 2. Mixed PD/ET resources requiring registration or an agreement

| Resource | Evidence for PD/ET content | Actual access boundary |
| --- | --- | --- |
| [B(RAIN)², DABI](https://dabi.loni.usc.edu/brain2/data) | First release 45 patients, second release 30 additional patients; **75 total**, all PD or ET, according to the [2025 descriptor](https://www.nature.com/articles/s41597-025-04784-5). Intracranial EEG, subcortical local field potentials, MRI/CT and clinical metadata in BIDS format. Separate PD/ET counts were not verified. | [Current terms](https://dabi.loni.usc.edu/brain2/privacy-policy-and-terms-of-use): sign a DUA for each release and provide a brief project description. Redistribution is prohibited. Release pages and terms were inspected while logged out; data were not accessed behind an account. |
| [PDBP, NIH/NINDS](https://pdbp.ninds.nih.gov/node/126) | University of Florida project explicitly includes PD and ET; historical enrollment goals include 30 for each diagnosis. DTI/fMRI, clinical assessments and biomarkers. Goals are not verified current downloadable counts. | **Controlled access**, not anonymous public download. [Current guide](https://pdbp.ninds.nih.gov/how-to-guide) requires a DUA co-signed by the PI and institutional signing official, an identity/account request and approval. |
| [PPMI](https://www.ppmi-info.org/access-data-specimens/data) | Public clinical-diagnosis forms include ET as a possible diagnosis. This makes PPMI a potential source for a carefully selected diagnostic subset; actual ET case counts and matched modality availability were not verified. | Registration, DUA and limited screening, per the [official FAQ](https://www.ppmi-info.org/help-and-resources/faqs). PD/HC/SWEDD cohort membership is not an ET label. **Do not treat all SWEDD participants as ET.** [Clinical form](https://www.ppmi-info.org/sites/default/files/docs/AM3PPMIClinicalCRFPacket20250806.pdf). |

## 3. Public additional candidates and small examples

| Source | What is actually public | Why it is separate from the confirmed classification shortlist |
| --- | --- | --- |
| [Sonication-level focused-ultrasound data, Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.qjq2bvqs7) | A 456.02-KB treatment-event CSV and README; CC0. README defines `diagnosis_ET0_PD1` as 0 = ET, 1 = PD. The anonymous CSV preview showed actual ET rows and subject/event identifiers. | Both code values are defined, but the full CSV's PD/ET counts and presence of PD rows were not confirmed. File-download checks failed; only the partial preview was inspected. Treatment power, temperature, targeting, and device settings are downstream of treatment decisions and are unsuitable as ordinary pretreatment diagnostic features. |
| [TIM-Tremor, 4TU](https://data.4tu.nl/articles/dataset/Technology_in_Motion_Tremor_Dataset_TIM-Tremor/12694256/1) | Described as public in the [original study](https://arxiv.org/abs/1809.03218): 55 people, RGB/depth video and accelerometry, 21 tasks. | Official repository was unavailable/under maintenance in this search. Current file manifests and data license were not verified. A [2024 analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC11098891/) reports 105 parkinsonian-tremor and 182 ET videos, not those numbers of independent people. Tremor phenotype and disease diagnosis must be distinguished. |
| [PD/ET demonstration video, Figshare](https://figshare.com/articles/media/Postural_and_Intention_Tremors_in_Essential_Tremor_and_Parkinson_s_Disease/678238) | One WMV, 61,466,455 bytes; 11 numbered clinical examples; CC BY 4.0. | Illustrative video, not a participant-indexed benchmark. Eleven examples do not establish eleven independent people. |
| [Tremoroton supplements](https://pmc.ncbi.nlm.nih.gov/articles/PMC6961062/) | Two approximately 1.7-MB TXT examples, a supplementary DOCX and an analysis PDF are publicly linked from the paper. The platform accepts bilateral acceleration and EMG. | The paper's validation cohort was 10 PD + 10 ET, but that does **not** establish release of all 20 people. Sample-file diagnosis labels were not verified. A supplementary download returned an HTML challenge; it was not accepted as a genuine DOCX. |

## 4. Additional anonymous public ET sources without a verified PD group

These sources can support ET modeling, additional exploratory data, or protocol design. Combining them with an unrelated PD repository can create disease/device/source confounding. They are not, by themselves, PD-versus-ET benchmarks. Counts refer to the indicated release or paper, not necessarily to inspected raw signal files.

| Source | Participants/labels | Modality and released files | Access/license and verification |
| --- | --- | --- | --- |
| [ET accelerometry, Pardo-Valencia et al. (2026), Zenodo](https://zenodo.org/records/19130599) | **29 ET**, 19 men/10 women | Both hands, rest/posture; acceleration at **5000 Hz**, nominal 100-second recordings; cleaned mean duration 73.4 seconds. `acc_signal_database.mat`, `clinical_data.mat`, README; ~90.1 MB | Current page shows Open and **CC BY 4.0**; description and file list inspected. Raw MATLAB file not loaded. |
| [Activity-based ET smartwatch data, Mendeley](https://data.mendeley.com/datasets/fd6sdp6t7x/1) | ET; participant count not verified | Pebble acceleration during Fahn–Tolosa–Marin assessments; `ET_Activity_Score_data.RData`, 3,387,853 bytes | Public file manifest; **CC BY 4.0**. Internal RData structure and acquisition frequency not verified. |
| [Phase-locked ET stimulation, Harvard Dataverse](https://doi.org/10.7910/DVN/Z6EN2I) | **18 unique ET**: 11 original + 7 new; repeat data for 6 original people | Acceleration, filtered signals and stimulation output; three MATLAB files and README | Files unrestricted; **CC0**. Actual README and manifest inspected. The six repeats are not six additional independent people. |
| [ET postural-perturbation/feedback dataset, Zenodo](https://zenodo.org/records/10843311) | **24 ET + 28 neurologically intact**; confirmed by counting participant CSV IDs | Kinarm calibration/perturbation trials and processed/normalized EMG from six muscles; clinical FTM-TRS scores; four ZIPs, README and participant CSV; **23.2 GB** | Open, **CC BY 4.0**; small README and CSV downloaded and inspected. [Published study](https://doi.org/10.1523/JNEUROSCI.0069-24.2024). |
| [ET saccadic/reaching adaptation, Zenodo](https://zenodo.org/records/8383213) | Study recruited **22 ET + 20 controls**; analyzed task cohorts typically **20 ET + 20 controls** after task-specific exclusions | Eye-movement and arm force-field adaptation; `DataAM01.zip`, **6.2 GB**; related supplement [8307950](https://doi.org/10.5281/zenodo.8307950) | Main release Open, **CC BY 4.0** in registered metadata. Archive not loaded. [Primary paper](https://journals.physiology.org/doi/full/10.1152/jn.00165.2023). |
| [ET intraoperative VIM/TAPS recordings, Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.cjsxksngp) | **9 ET** | Processed microelectrode signals, local field potentials, spikes, spectral measurements and clinical TETRAS scores; MATLAB/code; **77.82 GB** | Public files and detailed README; **CC0**. README/manifest inspected, archives not loaded. Processed files include a `raw` signal field, but the deposited package is a processed analysis release. |
| [Structural/diffusion MRI in ET, OpenNeuro ds008336](https://openneuro.org/datasets/ds008336/versions/1.0.0) | **12 participants: 9 `ET`, 3 `ETp`** in public `participants.tsv`; no PD | T1, T2-FLAIR, white-matter-nulled and diffusion MRI in BIDS; anatomical images defaced | **CC0**, confirmed in `dataset_description.json`. [Public repository](https://github.com/OpenNeuroDatasets/ds008336). Metadata inspected, MRI volumes not downloaded. Preserve the `ETp` distinction; do not silently relabel all 12 as pure ET. |
| [Cerebellar RNA-seq, GEO GSE134878](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE134878) | Public SOFT metadata: **54 samples**, including 32 ordinary ET, 20 ordinary control, one ET outlier and one control outlier | Human postmortem cerebellar RNA-seq; processed data and raw-sequence links | Public series and small SOFT metadata verified. Summary says 33 ET + 22 controls and two exclusions; the release metadata does not match that summary total. Use explicit sample IDs and outlier flags. Sequence files not downloaded. |
| [Cerebellar DNA methylation, GEO GSE134426](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE134426) | **12 ET + 11 controls**, confirmed from public SOFT sample metadata | Reduced-representation bisulfite sequencing/methylation | Public series and sample metadata verified; no PD group. Data files not loaded. No additional Creative Commons license was verified for GEO data. |
| [Blood gene counts, Zenodo](https://zenodo.org/records/21911278) / [GEO GSE344274](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE344274) | GEO release metadata: **35 ET + 29 controls = 64 samples**; study summary instead describes 38 ET + 37 controls | Public gene-level count matrices; Zenodo `gene_count.xls`, **16.9 MB** | Zenodo Open, **CC BY 4.0**. GEO sample metadata inspected. Raw sequencing is **controlled access** at [EGA EGAS50000002130](https://www.ega-archive.org/studies/EGAS50000002130). These repositories relate to the same study and must not be counted as independent cohorts. |
| [ET family whole-genome variants, Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.td8d20v) | **40 people in 8 ET families**, including affected/unaffected relatives; not 40 ET cases | 40 VCF.GZ variant-call files and sample manifest; **12.73 GB** | Public manifest; **CC0** in registered metadata. Sequence reads are not the released VCFs. Family membership creates dependence in classification splits. |
| [Cerebellar rTMS ET trial, Mendeley](https://data.mendeley.com/datasets/tt8j3bhy5w/1) | ET treatment trial; participant count not inspected | `Olfati-data.sav`, **11,442 bytes**, SPSS clinical/treatment data | Public file manifest; **CC BY 4.0**. File contents not inspected. |
| [ET median-nerve ultrasonography/electrophysiology, Figshare](https://doi.org/10.6084/m9.figshare.8028095) | ET; release's participant count and possible control rows not inspected | `pone.0215750.s001.sav`, **6,608 bytes**; individual clinical/nerve measurements | Public manifest; **CC BY 4.0**. Linked [primary study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0215750). Internal file contents not inspected. |
| [VIM stimulation and motor learning, Figshare](https://doi.org/10.6084/m9.figshare.25459117) | ET/DBS study; participant count not verified | Participant-level normalized learning scores in `BehavioralData.xlsx`; MATLAB **grand-average** spectral data and statistical results | Public manifest; **CC BY 4.0**. Spectral grand averages are not participant-level EEG recordings. |

The two Blondiaux releases, the Oxford/Harvard stimulation releases, and any repeated institution-specific cohorts may overlap. Their participant counts must not be summed without matching cohort provenance.

## 5. Exclusions: public papers, metadata or code do not establish public participant data

| Source | Why excluded from anonymous public PD/ET data |
| --- | --- |
| [Häring et al. multicenter accelerometry, 2025](https://doi.org/10.1002/mds.70032) | 226 PD + 188 ET, but individual data offered upon reasonable author request. |
| [Weede et al. Kiel accelerometry, 2024](https://doi.org/10.3390/info15040231) | 130 PD + 209 ET + 54 HC usable; no public release established; data-availability statement says sharing is not applicable. |
| [Czech PD/ET speech cohort, 2024](https://www.nature.com/articles/s41746-024-01027-6) | 50 PD + 50 ET + 50 HC; privacy-related reasonable-request access. No anonymous recording download. |
| [Cc-PhD handwriting, 2025](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0318021) | PD/ET/HC, but full dataset requires application and review. |
| [BIODARW / ET spirals](https://www.nature.com/articles/s41598-022-16741-y) | Participant data not publicly released because of privacy; analysis code is not the dataset. |
| [Hand-Drawn Image (HDI), 2026](https://link.springer.com/article/10.1007/s11042-026-21422-3) | 28 ET + 28 HC; primary article explicitly says data shared on request. |
| [PD/ET EMG prediction repository](https://github.com/larocs/EMG-prediction) | Repository describes a private EMG dataset; public code does not release patient recordings. |
| [PD/ET closed-loop TMS, Oxford](https://data.mrc.ox.ac.uk/data-set/emg-accelerometer-closed-loop-tms-et-and-pd) | Data upon request; may require a data-transfer agreement. |
| [Precision Imaging ET (PIET), ScienceDB](https://www.scidb.cn/en/detail?dataSetId=cf3e00f9dd514d22968f9e53a47f6a9a) | **Withdrawn from public access.** Current download section says hospital requirements prevent public access and directs inquiries to authors, despite a PUBLIC badge and CC0 metadata. |
| [ET thalamotomy microstructure MRI, Zenodo latest v2](https://zenodo.org/records/22249762) | Current repository explicitly says **Restricted**: public record, restricted files. |
| [STIL orthosis ET trial](https://doi.org/10.17026/ls/wv4asd) | Registered metadata specifies restricted access. |
| [Oxford ET LFP, three movement tasks](https://data.mrc.ox.ac.uk/lfp-et-dbs) | 8 ET; CC BY-SA 4.0, but current page says users must register/log in to download **or request** access. Anonymous download not available; exact post-login approval boundary unverified. |
| [Oxford naturalistic-movement EEG/OPM study](https://doi.org/10.60964/bndu-k5gj-9892) | Primary [study](https://publications.rwth-aachen.de/record/1032668/files/1032668.pdf) explicitly says raw data on request, with possible data-transfer agreement; public Zenodo release is analysis code. |
| [PD/ET gait classifier supplement 1](https://doi.org/10.6084/m9.figshare.12947846) and [supplement 2](https://doi.org/10.6084/m9.figshare.12947849) | Files are feature descriptions and model-performance tables, not the labeled gait recordings. |
| [PD/ET consumer accelerometry supplement](https://doi.org/10.6084/m9.figshare.12096141) | DOCX contains supplementary comparison figures, not the underlying time series. |
| [PD/ET cognition supplement](https://doi.org/10.6084/m9.figshare.29324343) | Actual DOCX inspected: bivariate correlation table for the study, not 188 PD/53 ET individual cognitive records. |
| [PD/ET smartphone spectral table](https://doi.org/10.6084/m9.figshare.5348449) | Published group-summary table, not a newly verified participant-level recording release. |
| [Longitudinal ET/dystonic tremor DBS, Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.380hc70) | Only a DOCX group-comparison supplementary table, despite a “dataset” repository label. |
| [Cerebellar ET stratification, Zenodo](https://doi.org/10.5281/zenodo.15541575) | Metadata describes MATLAB analysis codes; no verified patient-data release. |
| [Revisiting cognitive risks of ET, Harvard](https://doi.org/10.7910/DVN/ZVPGDV) | Public CSVs/plots relate to genetic-summary/reanalysis outputs; no verified individual PD/ET cohort. |
| [ALAMEDA PD tremor features](https://zenodo.org/records/10782573) | Open PD tremor-presence features, but no verified ET group. |
| [PhysioNet tremordb](https://physionet.org/content/tremordb/) | PD tremor data; ET must not be inferred from tremor alone. Repackaged feature datasets are not independent ET cohorts. |

### Additional metadata leads not promoted to confirmed data

The following indexed entries remain in the search inventory, but file contents/access or their diagnostic utility were not fully verified. They must not be treated as confirmed new public PD/ET cohorts:

- [Human Vim-DBS single-unit data, Mendeley](https://data.mendeley.com/datasets/ktrts7r2hk/1): description reports recordings from 19 ET patients and CC BY-ND 4.0; the checked public file-list endpoint returned an empty list.
- [NIRS–MRgFUS data](https://doi.org/10.5281/zenodo.7940314), [MRgFUS tractography measurements](https://doi.org/10.5281/zenodo.14809960), [MRgFUS pain measurements](https://doi.org/10.5281/zenodo.14795639), and [dystonic/ET targeting pilot](https://doi.org/10.5281/zenodo.10727257): treatment/imaging-related ET candidates, not verified PD/ET benchmarks.
- [ET diagnostic-marker supplement](https://doi.org/10.5281/zenodo.7971542): supplementary information, not confirmed as a new independent patient dataset.
- [BARET](https://doi.org/10.34760/69d60675243b5): brain-imaging metadata catalog; open files not verified.
- [DABI ET rest/movement/posture](https://doi.org/10.18120/chha-kv70), [coordinated-reset ET DBS](https://doi.org/10.18120/ts18-m231), [dual-lead thalamic DBS](https://doi.org/10.18120/hy35-pv20), and [responsive ET DBS](https://doi.org/10.18120/xj61-4a84): DABI projects, subject to their account/project access terms; no anonymous downloads verified.
- [ET NUS1 variant data](https://data.mendeley.com/datasets/zzbh2dtjvz/1): genetic-association supplement, not verified participant-level PD/ET data.
- [Handwritten PD/ET/control spirals, Wickramasinghe et al.](https://arxiv.org/abs/2006.12121): paper describes a large clinical image collection; no public participant-file release was established.
- [Chinese ET Registry](https://pd-mdcnc.com/zh-en/CETR-en.html): clinical/genetic cohort description, not a verified anonymous public release.
- [OpenSim wrist tremor model](https://doi.org/10.21227/715r-9b69): simulated/model data, not independently acquired PD/ET patient recordings.

## 6. Search methods, saved evidence and limits

The search combined repository indexing, primary-paper data-availability statements, article supplements, file manifests, README files, and diagnosis-label inspection. It covered wearable/accelerometry/EMG, handwriting/drawing, gait, speech, video, electrophysiology, imaging, clinical measurements, and molecular data.

Repository and index coverage included PhysioNet, OSF, Zenodo, Figshare/publisher Figshare portals, Mendeley Data, Harvard Dataverse, Dryad, OpenNeuro, Oxford ORA/BNDU, 4TU, ScienceDB, DABI, GEO/SRA/EGA, NIH PDBP, PPMI, and public author GitHub repositories. Targeted web searches included UCI/Kaggle alternatives; no additional verified mixed PD/ET cohort was established from those searches. Versions, mirrors and derived repackagings were not counted as independent participant cohorts.

### Explicit indexed queries

- DataCite API: `query="essential tremor"`, filter `resource-type-id=dataset`; **179 DOI records**, all two pages retrieved.
- DataCite API: `query=Parkinson* AND tremor`, same dataset filter; **277 DOI records**, all results retrieved.
- Union: **411 distinct DOI records**, including versions/concept DOIs; **206 distinct combined title strings**. These are metadata counts, not 411 independent patient datasets.
- Figshare public article search: `search_for="essential tremor"` with literal quotation marks for an exact phrase; six pages of 100, 100, 100, 100, 100 and 46 records: **546 entries**. All pages retrieved. Many entries are figures, article supplements or summaries. File manifests were inspected for the most plausible participant-data candidates, not every one of the 546 entries.
- OSF public title filter `tremor`: **29 nodes**, all pages retrieved; irrelevant nonclinical uses of “tremor” occur.
- Targeted web searches combined Parkinson/PD, essential tremor/ET, dataset/database/public/open/download/data availability with each modality; primary-paper references were followed where relevant.

### Local evidence

`data_catalog/systematic_search/` contains the original paginated responses, manifests, small downloaded supplementary tables, public SOFT metadata, and inspection summaries. Key indexes:

- `search_summary.json`: query counts and limitations.
- `datacite_inventory.tsv`: 411 DOI metadata records, source URLs, descriptions and rights.
- `figshare_inventory.tsv`: all 546 exact-phrase results and whether a full manifest was inspected.
- `digital_assessment.xlsx` and parsed sheet previews: actual PD/ET spiral measurements.
- `motor_plasticity.xls` and `motor_plasticity_structure.json`: actual PD/ET/HC measurements.
- `serotonin_clinical.txt`, `serotonin_csv.txt`, repository tree and primary article: neurochemical study.
- `et_feedback_readme.txt` and `et_feedback_participants.csv`: 24 ET + 28 controls.
- `openneuro_participants.decoded.txt` and `openneuro_description.decoded.txt`: 9 ET, 3 ETp, CC0.
- `GSE134878_family.soft`, `GSE134426_family.soft`, `GSE344274_family.soft`: actual released sample metadata.
- Earlier verified wearable evidence remains in `data_catalog/verification.json`, `pads_participants.csv`, `cologne_training_postural_annotations.csv` and `cologne_license.txt`.

Some HTTP requests returned access challenges or errors. The HTTP success flag in the small-fetch log records transport success only; an HTML challenge was not accepted as a valid supplementary file. Full archives were not downloaded merely to prove that their listings exist. No accounts were created, DUAs accepted, or authors contacted.

This is a systematic search with explicit coverage and evidence, **not proof that no other dataset exists**. Unindexed repositories, local-language descriptions, supplementary files without searchable metadata and subsequent releases can escape the search. Remaining candidates and inaccessible files are reported explicitly rather than silently counted as open.

## 7. Recommended choice for a PD/ET classification study

For a noninvasive motion-sensor classifier, begin with **PADS**, then assess **Cologne** as a distinct cohort while preserving device/task/treatment differences. Use **Arm Tremor** primarily for prototyping because it contains only two ET participants. For a small feature-based study, the **Lin spiral workbook** and **Lu cortical/DTI workbook** provide additional openly inspectable diagnoses and participant measurements. The **caudate** source and **B(RAIN)²** suit specialized invasive/neuroimaging research.

Split by person before creating windows or selecting trials. Keep both hands, repeated tasks, recording sessions and relevant related-family samples in the same evaluation group. Fit preprocessing/model selection only on training data. Report participant denominators and class-specific outcomes; more windows do not create more independent participants. Do not add participant totals across repositories before checking cohort overlap.
