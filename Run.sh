
#path to file - to pass from diff dir. change variable
file_pathOK=OK
file_pathWB=WB

echo "preprocessing dataset..."
for path in $file_pathOK $file_pathWB; do python Preprocess_dataset.py $path; done;
echo "DONE DATA PREPROCESSING!"
echo "Classifying..."
python Classify.py
