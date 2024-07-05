[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/93Kh-MH9)

# Data Analysis
We began by conducting checks on the provided dataset's integrity, followed by an in-depth analysis of the data. We employed visual aids in the form of graphs to effectively present the data's distribution and inherent characteristics.

# K-Nearest Neighbors (KNN) Strategies
We devised two distinct renditions of the K-Nearest Neighbors algorithm:

Numpy-Optimized Version: To boost performance, we optimized the algorithm using numpy arrays for streamlined vectorized operations. This adjustment significantly expedited processing by capitalizing on parallel computing capabilities.

SKLearn KNN Integration: Our third avenue involved integrating SKLearn's KNN implementation—a prevalent machine learning framework.

Our optimization strategy focused on leveraging vector operations, such as numpy-driven vector multiplication, to accelerate computations in contrast to the initial loop-based implementation.

Concerning the fine-tuning of the model, we conducted experiments encompassing diverse hyperparameters:

Model Configuration Experimentation: We ran the optimized model through 42 hyperparameter combinations:
k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
encoders = ['vit', 'res']
distance_metrics = ['l1', 'cosine', 'l2']
The primary objective was to pinpoint the configuration yielding the highest accuracy. Following meticulous experimentation, we ascertained the optimal configuration: Encoder - 'vit', Distance Metric - 'l1', k - 7.

Graphical depictions aided in visualizing the hyperparameter tuning process, facilitating the identification of the most suitable configuration. Additionally, we plotted a graph showcasing the relationship between Inference Time and Data Split.

Our evaluation metrics encompassed SKLearn Accuracy, F1-Score, Precision, and Recall.

# Decision Tree Approaches
## Power Set Approach
Addressing the multilabel challenge involving 8 unique labels, we adopted a Power Set strategy:

We generated a power set comprising 2^8 elements.
Transformation of multilabels into power set indices was undertaken. Alternatively, one-hot encoding could have been employed, but using power set indices preserved label-specific information.
A singular Decision Tree was implemented to tackle the multiclass issue. Our evaluation metrics spanned SKLearn Accuracy, Macro & Micro F1-Score, Precision, and Recall.

# MultiOutput Strategy
We also embraced a MultiOutput methodology:

Multiple Decision Trees were crafted, each devoted to predicting a distinct label feature (8 Decision Trees in this instance, corresponding to the 8 unique labels).
Accuracy was gauged via the Hamming Loss, which evaluated predictions for each label individually, acknowledging partially accurate predictions.
This approach facilitated a nuanced evaluation, wherein even partially accurate label predictions were acknowledged. Unlike traditional accuracy metrics, which demand perfect label matches, this approach recognized the granularity of predictions.

The preceding documentation outlines the phases of data analysis, KNN implementation strategies with optimization, the hyperparameter tuning process, the methodologies for implementing decision trees, and the range of evaluation metrics leveraged for each model. Visual representations were created to provide insights into various dimensions of these processes.