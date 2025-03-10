# Study of children’s emotional expressions and their relationship with the use of colour in digital drawings 
This repository contanins the code used in the paper <em>Study of children’s emotional expressions and their relationship with the use of colour in digital drawings </em>.

## Summary
The work studies audios and drawings of pre-school children using three emotional expressions (‘Positive’, ‘Negative’ and ‘Neutral’).

Subsequently, in the framework of a free drawing activity controlled by their regular teachers, these classified audios are used to obtain the consensus reached with the teachers' labelling. Finally, the drawings associated with each audio are analysed using the labelling of the expert (teachers).

In these drawings, the use of colour is analysed and the differences between the different emotional expressions are examined for our case study.

## Machine Learning algorithms
We have used several algorithms to perform audio classification: Multilayer Perceptron (MLP), Support Vector Machine (SVM) with different kernels, K Nearest Neighbors (KNN) and Logistic Regression (LR). Additionally, we have produced three meta-models using ensemble stacking. 

All of them have been implemented in Python.

## Datasets
Three datasets have been used, which are the following: <em>Mexican Emotional Speech Database</em> (MESD) and our case study (Draw&Talk). It should be noted that the first one is audio-only, being used for training, and the second one includes audio files and drawings.

The only dataset available online is MESD (via the Kaggle website under the Attribution 4.0 International (CC BY 4.0) licence). Draw&Talk should be requested from the authors of this article.
