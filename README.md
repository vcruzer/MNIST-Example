# MNIST Example

Hello friend ! This repository contains simplistic implementation for the MNIST dataset. It uses the same CNN networks for Torch and Tensorflow (Keras) and XGBoost.

It also contains a hyperparameter search in *hyperparam_opt.py*. The search for XGBoost with the help of SKLEARN (K-fold cross validation). Torch uses RayTune with Random search. Keras uses KerasTuner with random search as well. Their best parameters are saved inside the params folder as .json dictionary files.