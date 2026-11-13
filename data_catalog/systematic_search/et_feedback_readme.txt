Dataset Description
Files:

    Participant Data:
        * One calibration matrix per participant
        * One trials matrix per participant
        * Additional participant information such as age, gender, and FTM-TRS score in a separate CSV file.

Calibration Trials:

    * Participants performed calibration trials by applying forces on a locked handle at the beginning, middle, and end of the experiment.
    * Interleaved with the experimental blocks from the trials matrix.
    * Each calibration trial consisted of applying forces in four directions, three times, resulting in 12 trials per block and a total of 36 trials.
    * Data structure follows the format specified by Kinarm; refer to their documentation for a description of each field.
    * File names indicate the direction and repetition number of each trial.
    * EMG data includes filtered and normalized readings from muscles in the following order: Pectoralis, Deltoid, Brachialis, Biceps, Lateral head triceps, and Long head triceps.
    * Calibration trials are only necessary if one's wants to recompute the normalization.

Trials:

    * Participants' data during the postural perturbation task.
    * Each block consists of 60 trials, repeated five times, resulting in a total of 300 trials.
    * Trial types range from 1 to 12, representing four directions and three forces.
    * Trial field in the file names indicates the trial type, aiding in identifying the repetition and trial type.
    * EMG data in these files is also filtered and normalized.
    * Offline computed indices such as start_t, stop_t, and crossed_t are included, see methods for details about their computation. 