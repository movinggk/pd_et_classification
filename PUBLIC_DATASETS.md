# Public datasets for Parkinson's disease versus essential tremor classification

Checked on 6 October 2026. PD means Parkinson's disease; ET means essential tremor. This catalog concerns classification between these diagnoses, rather than PD versus healthy controls or tremor severity alone. The original search downloaded metadata but no full signal dataset. No model was trained.

**PADS acquisition update, 7 October 2026:** the complete v1.0.0 release is now available in [`data/pads/1.0.0/`](data/pads/1.0.0/). All 12,689 files in the official checksum list passed SHA-256 verification. See the [acquisition guide](data/pads/README.md) for the local data layout, validation results, and the 304-participant PD/ET index.

**Expanded systematic search:** see [PUBLIC_DATA_SEARCH.md](PUBLIC_DATA_SEARCH.md) for six confirmed anonymous PD/ET data sources, registered-access resources, additional ET-only datasets, explicit exclusions and the saved search inventory. That report expands the original wearable-focused shortlist below.

## Shortlist

| Dataset | Participants relevant to PD versus ET | Measurements | Access and license | Assessment |
| --- | --- | --- | --- | --- |
| [PADS, PhysioNet v1.0.0](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/) | 276 PD and 28 ET; 469 participants overall | Bilateral wrist acceleration and gyroscope signals at 100 Hz, 11 clinical movement tasks, nonmotor questionnaire, demographics | Open download; CC BY-NC-SA 4.0; ZIP 735.0 MB, uncompressed 1.4 GB | Recommended starting point for a wearable-sensor PD-versus-ET study, based on the verified cohort size and explicit diagnosis labels |
| [Cologne accelerometry dataset, OSF](https://osf.io/mp5ha/) | Paper reports 25 PD and 16 ET participants, 54 upper limbs, repeated assessments | Wrist accelerometry at 100 Hz and finger accelerometry at 2500 Hz; resting and postural trials; raw CSV and clinical ratings | Public repository and downloads verified; the repository license file specifies CC BY-NC-SA 4.0 | Useful additional PD/ET cohort; all participants were treated with deep brain stimulation (DBS) |
| [Arm Tremor Database, Zenodo version 2](https://zenodo.org/records/4772130) | 8 PD, 2 ET, 27 participants with no diagnosed tremor | Four triaxial accelerometers at 62.5 Hz; four movement conditions; MATLAB file | Open download; CC BY 4.0; database.mat is 6,749,196 bytes | Useful for pipeline prototyping; only two ET participants |
| [TIM-Tremor, 4TU.ResearchData](https://data.4tu.nl/articles/dataset/Technology_in_Motion_Tremor_Dataset_TIM-Tremor/12694256/1) | Published work describes 55 participants; a 2024 analysis reports 105 parkinsonian-tremor and 182 ET videos | RGB video, depth video and accelerometry, with clinical tremor annotations | Described as public in published research; the official repository returned a maintenance page during this check; current license and downloads remain unverified | Candidate for video/multimodal work; video counts are not participant counts |

## PADS: diagnosis selection and limitations

I downloaded and counted the official [preprocessed/file_list.csv](https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/preprocessed/file_list.csv). The condition counts are 276 `Parkinson's`, 28 `Essential Tremor`, 79 `Healthy`, 15 `Atypical Parkinsonism`, 11 `Multiple Sclerosis`, and 60 `Other Movement Disorders`.

For a binary study using the published diagnosis assignments, select `condition == "Parkinson's"` or `condition == "Essential Tremor"`. This yields 304 participants before any additional exclusions. The numeric `label` column uses 0 for healthy, 1 for PD, and 2 for the combined differential-diagnosis group. Label 2 alone does not identify ET.

Some PD-assigned records mention ET in `disease_comment`. These comments are not proof of a particular alternative diagnosis; they require a predefined review policy if the intended study excludes overlapping or uncertain diagnoses. Do not silently assign diagnoses by keyword searches in comments.

Join recordings to participants using the participant ID. The dataset includes raw time series, patient JSON, questionnaire JSON and preprocessed files. The dataset authors recommend removing approximately the first 0.5 seconds of each raw recording because the watch gives a vibration notification. There is substantial PD/ET class imbalance.

Links: [dataset and download instructions](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/), [participant metadata](https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/preprocessed/file_list.csv), [original study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10770131/).

## Cologne: labels and repository license discrepancy

The downloaded `TrainingData_PosturalTremor.csv` contains `Measurement`, `SubjectID`, `Disease`, `Side`, `PosturalTremor`, `FingerAccelerometry`, and `WristAccelerometry`. Its `Disease` column explicitly uses ET and PD, while `PosturalTremor` is a separate severity target. This table has 478 rows: 234 ET trial rows and 244 PD trial rows. These are repeated trial counts, not independent participant counts. Missing recording links occur in this table.

The repository has separate raw finger/wrist recording folders for training and validation, four resting/postural annotation tables, and scripts. Its raw finger recording index was checked through the public OSF API.

The [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10611060/) reports CC BY-SA 4.0. However, the actual [repository license file](https://osf.io/download/z9t6d/) specifies Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). This catalog reports the downloadable file's more restrictive terms. The paper's original split is described at the upper-limb level; a participant-independent diagnostic experiment must keep both limbs and all repeated trials from the same person in one fold.

Links: [repository](https://osf.io/mp5ha/), [training postural annotations](https://osf.io/download/46b3p/), [training resting annotations](https://osf.io/download/kg9xm/), [validation postural annotations](https://osf.io/download/yq275/), [validation resting annotations](https://osf.io/download/u5r6s/), [license](https://osf.io/download/z9t6d/).

## Arm Tremor Database: exact structure

The [README](https://zenodo.org/records/4772130/files/readme.txt?download=1) describes a `data` cell vector of 37 participant structs: entries 1-2 are ET, entries 3-10 are PD, and entries 11-37 have no diagnosed tremor. Each struct contains participant information and recordings. Four conditions are included: rest, postural, free motion, and motion with an object. Sensor sites are the index finger, thumb, metacarpal region, and wrist. Each sensor supplies x/y/z acceleration in m/s² at 62.5 Hz. Medication status is recorded.

The cohort counts and structure were verified from the README; the MATLAB signal file itself was not inspected. Open access, the CC BY 4.0 license, file size and published checksum were verified from Zenodo metadata.

Links: [dataset](https://zenodo.org/records/4772130), [database.mat](https://zenodo.org/records/4772130/files/database.mat?download=1), [README](https://zenodo.org/records/4772130/files/readme.txt?download=1), [study](https://doi.org/10.3390/signals2020016).

## TIM-Tremor and other modalities

[The original TIM-Tremor study](https://arxiv.org/abs/1809.03218) describes video/depth and accelerometer data for 55 patients. [A later study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11098891/) reports parkinsonian, essential, functional and dystonic tremor video classes. Counts refer to videos, not unique patients, and tremor-type classification must be distinguished from independently established disease diagnosis. The repository was offline for maintenance when checked; its current label files and reuse terms were not verified.

For handwriting, Cc-PhD includes PD, ET and healthy controls, but its [2025 PLOS paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0318021) explicitly states that the full dataset is not publicly available; access requires an application and review. It is therefore not an immediately open-download option.

[ALAMEDA](https://zenodo.org/records/10782573) is openly downloadable but contains PD tremor-presence labels and extracted accelerometer features; it does not provide an ET comparison cohort.

## Additional sources found in the expanded search

The original shortlist is not exhaustive. The sources below distinguish datasets containing both diagnoses from ET-only supplementary data and from clinical demonstration videos. An open-access article does not establish that its participant data are open.

| Additional source | Cohort or material | Measurements | Verified access | Relevance |
| --- | --- | --- | --- | --- |
| [Häring et al., multicenter tremor cohort (2025)](https://movementdisorders.onlinelibrary.wiley.com/doi/10.1002/mds.70032) | 226 PD and 188 ET; 414 participants across six centers | Hand accelerometry at rest and during posture; device and position vary by center; analysis uses 15-second segments downsampled to 100 Hz | Article explicitly offers data from the corresponding author upon reasonable request; no open participant-data download was verified | Large additional PD/ET candidate, subject to obtaining access and its reuse terms |
| [Weede et al., Kiel tremor cohort (2024)](https://doi.org/10.3390/info15040231) | 130 PD, 209 ET, 54 healthy controls; 393 usable participants out of 408 examined | Bilateral uniaxial hand accelerometry at 800 Hz; three 30-second conditions | No public release confirmed; the article's data availability statement says data sharing is not applicable | Detailed protocol reference for collecting new data; not an established downloadable dataset |
| [Rios-Urrego et al., Czech PD/ET speech cohort (2024)](https://www.nature.com/articles/s41746-024-01027-6) | 50 PD, 50 ET, 50 healthy controls | Rapid /pa-ta-ka/ syllable repetitions and spontaneous monologue; recordings downsampled to 16 kHz | Individual participant data available on reasonable request; not publicly available because of participant privacy | PD/ET classification using speech rather than hand motion |
| [Activity-based ET smartwatch dataset, Mendeley Data (2018)](https://data.mendeley.com/datasets/fd6sdp6t7x/1) | ET data; participant count not verified; no PD group described | Pebble smartwatch acceleration during Fahn–Tolosa–Marin tremor assessment | Public file listing verified: ET_Activity_Score_data.RData, 3,387,853 bytes; CC BY 4.0 | Supplementary ET data and a collection-method reference; not a standalone PD/ET dataset |
| [Schreglmann et al., phase-locked ET stimulation replication data (2021)](https://doi.org/10.7910/DVN/Z6EN2I) | 11 original ET participants, 7 new ET participants, and repeated sessions from 6 of the original participants; 18 unique participants | Tremor acceleration recordings during a stimulation experiment; raw and filtered tremor signals and stimulation output | Harvard Dataverse metadata confirms three unrestricted MATLAB files and a README; CC0 1.0 | Additional ET signals; no PD comparison group; stimulation conditions must be considered |
| [Sternberg et al., PD/ET clinical demonstration video (2013)](https://figshare.com/articles/media/Postural_and_Intention_Tremors_in_Essential_Tremor_and_Parkinson_s_Disease/678238) | A video with 11 numbered PD/ET clinical examples; unique participant count not established | Postural and intention tremor demonstrations | Figshare API lists one WMV file, 61,466,455 bytes; CC BY 4.0 | Useful for understanding examination positions; a small illustrative collection, not a participant-indexed training benchmark |

### Notes for acquiring your own data

The Kiel study describes a seated resting condition with forearms supported and hands hanging comfortably, an arms-extended condition with hands parallel to the floor, and the same extended condition with a 1 kg weight attached to the back of the hand. Each condition lasts 30 seconds, with rest periods between them. Both hands are recorded synchronously. This is a published protocol description, not a personalized clinical recommendation. Its [full primary paper](https://mdpi-res.com/d_attachment/information/information-15-00231/article_deploy/information-15-00231.pdf?version=1713447535) provides the acquisition details.

The multicenter study records bilateral rest and posture, using monoaxial sensors in Kiel and triaxial sensors elsewhere. Its 100 Hz rate and 15-second duration describe the standardized analysis input, rather than a uniform original acquisition protocol. The historical Kiel component contains 99 PD and 99 ET participants. Independence from other Kiel releases or studies has not been established, so cohort totals from these papers must not be added without checking participant overlap.

The Czech speech study requests at least seven rapid /pa-ta-ka/ repetitions in one breath and recommends approximately 90 seconds of spontaneous speech without imposing a strict time limit. All PD patients were recorded in their medication ON state. These details may be useful if speech will be part of a newly collected PD/ET cohort.

The Harvard README explicitly distinguishes new participants from repeated sessions. Do not count its 6 repeat participants as additional independent people. The README names its repeat file slightly differently from the actual repository filename; the public file index lists `data_sbj_repeated_2019.mat`.

Combining ET-only data with PD data from another repository can create a disease/device/source confound: a classifier might learn which device or protocol produced a recording. These sources are useful for exploratory work, but such a combination does not establish a controlled PD-versus-ET benchmark.

Small public repository metadata and the Harvard README are saved in `data_catalog/expanded_verification.json` and `data_catalog/harvard_et_readme.txt`. The additional signal and video files themselves were not downloaded or inspected. Mendeley's participant count, sampling rate and internal RData structure remain unverified.

## Evaluation requirements

1. Define the target explicitly: PD versus ET diagnoses, or parkinsonian versus essential tremor types.
2. Split by participant before dividing signals into windows. Keep both hands, tasks and repeat trials of one participant in one fold.
3. Fit normalization, feature selection, resampling and model selection using training data only.
4. Report participant counts, class-specific sensitivity, balanced accuracy, macro-F1 and participant-level uncertainty. A large window count does not increase the number of independent participants.
5. When using a second dataset, retain differences in devices, sampling rates, tasks and treatment status as limitations of external validation.

Small verification files from the original search are saved in `data_catalog/`. They contain public annotations and the downloaded Cologne license, with sources recorded in `verification.json`. Full raw signal/video archives were not downloaded during that search; the later complete PADS acquisition is documented in [`data/pads/README.md`](data/pads/README.md).
