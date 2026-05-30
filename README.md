**If you see 2 files with the same name, use the file that has a '2' in the name**

**testtrain.py is recommended to run after training or running train_isolation_forest.py to check for training quality**
-ideal output: anomalies detected evenly across multiple sessions (a session means 1 set of exercise that contains multiple reps)
-bad output: anomalies detected cluster mainly in specific session(s)

**testromm.py is a guide to brief check the approximate rom before cleaning and interpolation of the recorded rep dataset, recommended to run after recording one rep if feeling unsure abt the angle**

**After done recording a good session with multiple good form reps with different speed, run filter_bad_reps.py, then extract_featrures.py to get the extracted cleaned dataset. After that, run train_isolation_forest.py**
