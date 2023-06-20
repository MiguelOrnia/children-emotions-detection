# Detecting emotional expressions in children at the pre-school stage and their relation with the use of color
This repository contanins the code used in the paper <em>Detecting emotional expressions in children at the pre-school stage and their relation with the use of color</em>.

## Summary
The paper classifies audios of pre-school children into three emotional expressions ("Positive", "Negative" and "Neutral").  

Subsequently, within the framework of a free drawing activity controlled by their regular teachers, these labelled audios are used to analyse the drawings associated with them.

On these drawings, the use of colour is analysed and the differences between the several emotional expressions are examined for our case study.

## Machine Learning algorithms
We have used three different algorithms to perform audio classification: CART Decision Tree (DT), Multilayer Perceptron (MLP) and Support Vector Machine (SVM).

All of them have been implemented in Python.

## Datasets
Three datasets have been used, which are the following: <em>Mexican Emotional Speech Database</em> (MESD), <em>Interactive Emotional Children's Speech Corpus</em> (IESC-Child) and our case study (Draw&Talk). It should be noted that the first two are audio-only, being used for training, and the third includes audio and drawings.

The only dataset available online is MESD (via the Kaggle website under the Attribution 4.0 International (CC BY 4.0) licence). The IESC-Child dataset should be requested from its authors and Draw&Talk should be requested from the authors of this paper.
