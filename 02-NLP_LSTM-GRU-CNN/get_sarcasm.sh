cd ./datasets/sarcasm/

# activate the environment
conda activate tf2.5

# training_cleaned.csv
wget --check-certificate \ https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json \ -O ./sarcasm.json
   
cd ../../

deactivate