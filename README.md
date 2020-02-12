# Will_my_bug_report_be_developed
This project uses a artificial neural network model to predict whether a bug report will be developed or not.

* The bug management system of Stack Exchange is used due to richness in features (like Q&A, reporter, gamification features etc.)

* Text features using NLP are generated from a Word2Vector word embedding model, these features are used along with the Stack Exchange specific features. 

Process:
- The data dump for Meta Stack Exchange is to be downloaded from archive.org 
- This gives us the csv files- posts.cvs, votes.csv, users.csv, etc. 
- The python module, Crawler.py collects the bug reports, attaches the reuiquired bug related infromation to the bug posts.

- The python module, StackExchangeFeatures.py collects all the Stack Exchange specific features. This includes extraction of Q&A features (Question Count, Answer Count, Comment Count etc.); reporter features (All previous bug reports posted, etc.); and gamification features (e.g., reputation)

- The python file WordEmbeddingFeatures.py converts the text of the bug report into word embedding vectors (doc2vec). The stack exchange features are also added.

- The jupyter notebook flie, DeepLearing.ipynb builds an Artificial Neural Network that classifies the developed and non-developed bugs
The performance of the ANN model is also compared  to a baseline machine learning model using random forests.
