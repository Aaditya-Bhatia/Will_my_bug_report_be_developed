# Will_my_bug_report_be_developed
Features are extracted from Stack Exchange bug reports, and a deep learning model is built to classify whether or not a developer would develop the bug report.


- The data dump for Stack Exchange is to be downloaded from archive.org
- This gives us the csv files- posts.cvs, votes.csv, users.csv, etc. 
- The python file getBugs.py collects the bug reports, attaches the reuiquired bug related infromation to the bug posts.
- The python file, getBugEdits.py collects all edits made on bug reports.
- The python file getBugFeatures.py attaches the different features of the bug reporters, bug report text, etc.
- The python file Preprocess_DeepLearning.py converts the text of the bug report into vectors (doc2vec) and attaches the other independent variables with the doc2vec variables.
- The jupyter notebook flie, Implementation_DeepLearing.ipynb performs model building with Artificial Neural Network and classifies the developed and non-developed bugs
The ANN model is also compared with respect to a baseline Random Forest Classifier model.
