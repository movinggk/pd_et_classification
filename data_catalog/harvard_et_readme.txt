The source data folder contains 3 ‘.mat’ files (MATLAB format data);

1. “data_sbj_original_2016.mat”: This is the data from 2016 original experiments, which contains data from 11 subjects;
2. “data_sbj_repeat_2019.mat”: This contains data for repeated experiments in 2019, which have 6 subjects from the original 11 subject cohort;
3. “data_sbj_new_2019.mat”: This contains data for experiments done on 7 new subject in 2019;

-------------------------------------------------------------------------------------------------

Data structure:
Each '.mat' file consists of 2 parts, 'data_all_res' and 'num_trial';

1. 'data_all_res' is a cell type variable, where the number of cells corresponds to the number of subjects. In each cell, there are 8 internal cells, where each cell represents a stimulation condition: 0 degree phase-locking, 60 degree, 120 degree, 180 degree, 240 degree, 300 degree, sham and non-phase-locking stimulation. Then within each condition cell, there will be several (inequivalent) trials, each trial represents a recording data. In the recording data structure, there are several fields, whose definition are listed below:

dt_us –time interval between data consecutive points in microseconds.

ADC_in_raw - time series data of the tremor (raw data), recorded via analog to digital converter (ADC) with a range of +/- 2048.

ADC_in – time series data of the tremor (bandpass filtered at the tremor frequency), recorded via analog to digital converter (ADC) with a range of +/- 2048. The filter that was uses is a 2nd order Butterworth bandpass with a cut-off frequencies f1= cal_freq - cal_freq/2 and f2= cal_freq + cal_freq/2.

DAC_out – time series data of the electrical stimulation output (detrended data), generated via digital to analog converter (DAC) with a range of 1-4096 which was detrended to +/- 2048.
t – time vector in seconds, created using the dt_mean (below).

dt_mean - mean time interval in seconds between consecutive data points. 

dt_std – standard deviation in seconds of the mean time interval between consecutive data points.

cal_freq – the mean tremor frequency in Hz that was computed during the short calibration period before the trial. The cal_freq frequency was used by real-time algorithm to compute the cut-off frequencies of the online bandpass filter prior to the endpoint corrected Hilbert transform. Note that the filter was used internally by the algorithm, i.e., the tremor data that was logged was not filtered. 

ADC_in_phase – the instantaneous phase of the ADC_in data computed postprocessing using via Hilbert transformation. 

DAC_out_phase - the instantaneous phase of the DAC_out data computed postprocessing using via Hilbert transformation.

ADC_in_amp - the instantaneous amplitude of the ADC_in data computed postprocessing using via Hilbert transformation.

DAC_out_amp - the instantaneous amplitude of the DAC_out data computed postprocessing using via Hilbert transformation.

ori - the original condition of this trial (before phase-binning and rearranging). For example, [2,5] means that this trial originally belongs to the 2nd repetition of condition No.5.

%%
For example, 'data_all_res{1}{3}{2}.ADC_in' takes the filtered tremor time series data from the 2nd repetition of the condition No.3 in subject 1.

--------------------

2. 'num_trial' is simply a Nx8 matrix (N is the number of subjects) containing the counts of number of trials in each condition for each subject.


