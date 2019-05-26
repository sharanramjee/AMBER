autoencoder_diagram.png: Illustration of the Autoencoder architecture
overfitting_demo_data.xlsx: Data corresponding to the toy experiment of Section 5.2 explaining that overfitting is useful for feature selection
dataset_links.txt: Text file containing links to download the 4 datasets
library_list.txt: List of all used External Libraries


Source Code Files (Source Code Folder):

(MNIST Sub-Folder)
mnist_fisher.py: Source code for using Fisher Score with the MNIST Digit Dataset
mnist_cmim.pp: Source code for using Conditional Mutual Information Maximization with the MNIST Digit Dataset
mnist_rfs.py: Source code for using Robust Feature Extraction with the MNIST Digit Dataset
mnist_fqi.py: Source code for using Feature Quality Index with the MNIST Digit Dataset
mnist_amber.py: Source code for using AMBER with the MNIST Digit Dataset

(Reuters Sub-Folder)
reuters_fisher.py: Source code for using Fisher Score with the Reuters Dataset
reuters_cmim.pp: Source code for using Conditional Mutual Information Maximization with the Reuters Dataset
reuters_rfs.py: Source code for using Robust Feature Extraction with the Reuters Dataset
reuters_fqi.py: Source code for using Feature Quality Index with the Reuters Dataset
reuters_amber.py: Source code for using AMBER with the Reuters Dataset

(Cancer Sub-Folder)
cancer_fisher.py: Source code for using Fisher Score with the Winconsin Breast Cancer Dataset
cancer_cmim.pp: Source code for using Conditional Mutual Information Maximization with the Winconsin Breast Cancer Dataset
cancer_rfs.py: Source code for using Robust Feature Extraction with the Winconsin Breast Cancer Dataset
cancer_fqi.py: Source code for using Feature Quality Index with the Winconsin Breast Cancer Dataset
cancer_amber.py: Source code for using AMBER with the Winconsin Breast Cancer Dataset

(RadioML Sub-Folder)
radioml_fisher.py: Source code for using Fisher Score with the RadioML2016.10b Dataset
radioml_cmim.pp: Source code for using Conditional Mutual Information Maximization with the RadioML2016.10b Dataset
radioml_rfs.py: Source code for using Robust Feature Extraction with the RadioML2016.10b Dataset
radioml_fqi.py: Source code for using Feature Quality Index with the RadioML2016.10b Dataset
radioml_amber.py: Source code for using AMBER with the RadioML2016.10b Dataset

The model subfolder in each of these folders contain the state-of-the-art ranker model used by AMBER.
The features subfolder in each of these folders are the destinations where the selected features are saved after running the codes so that they can be used in the future.

Classification Accuracy Files (Results Folder): 

mnist_data.xlsx: Obtained accuracy results for each the 3 runs and their average for the MNIST Digit Dataset
reuters_data.xlsx: Obtained accuracy results for each the 3 runs and their average for the Reuters Dataset
cancer_data.xlsx: Obtained accuracy results for each the 3 runs and their average for the Wisconsin Breast Cancer Dataset
radioml_data.xlsx: Obtained accuracy results for each the 3 runs and their average for the RadioML2016.10b Dataset


Classification Accuracy Curves with Error Bars (Error Bars Folder):

mnist_error_bars.png: Obtained accuracy curves versus feature count with error bars for the MNIST Digit Dataset
reuters_error_bars.png: Obtained accuracy curves versus feature count with error bars for the Reuters Dataset
cancer_error_bars.png: Obtained accuracy curves versus feature count with error bars for the Wisconsin Breast Cancer Dataset
radioml_error_bars.png: Obtained accuracy curves versus feature count with error bars for the RadioML2016.10b Dataset