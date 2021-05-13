# Deep neural network for predicting malicious URLs

### Usage
```
pip3 install -r requirements.txt

python3 fproj.py
```

The python script will begin by importing the malicious_phish.csv file from here:
https://drive.google.com/file/d/11aF3BWiQLhMSRAB6-UbKJap-OopCDa_E/view?usp=sharing
(Credit for compiling the dataset goes to Manu Siddhartha (https://www.kaggle.com/sid321axn/malicious))

The script will then extract lexical features from the sample and train the model.

Once training is complete, the script waits for the user to input URLs for prediction.

### Future Features:
* extend lexical feature set
* automate URL input
* direct results to a database