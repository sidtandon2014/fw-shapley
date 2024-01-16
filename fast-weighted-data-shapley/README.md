## Files
- ```model.py```: stores the architecture of the weighted shapley estimator network
- ```utils.py```: stores classes to compute utility (using KNN surrogate)
- ```train.py```: *ParameterizedShapleyEstimator* class (adaptation of FastSHAP class)
- ```extract_features.py``` - reduces the dimensionality of image datasets to 32 (features from ResNet18, then PCA). (features stored in ```/features/``` folder)


## Next
- FastSHAP like class for our case (Sid) [**DONE**]
- weighted shapley estimator model (Pranoy) [**DONE**]
- Dataset and Experimentation Pipeline (start with inclusion/exclusion and noisy data detection) (Sid)
- Utility Function (Pranoy) [**DONE**] (Have to perform some unit testing)
- Email authors of "Scalability and Utility" paper for getting the values used in the plots in their paper (Pranoy) [**DONE**]


## Issues
- FastSHAP uses grand val in normalization as well as in loss function but as per algo it should be sloghtly different. Check the algo. For the time being I am considering different grand for normalization and loss
- We need to come up with approximation for shapley kernel multiplier as it is combinatorial and might give arithmentic overflow error
- Need to write KNNUtility in batch fashion
- feature_extractor is taking last layer from Resnet (i.e. 1000 class output for imagenet)
- Needs discussion on loss regularization and normalization
- For large N weight list distribution is getting distorted (Pranoy & Sid)


## Experiments
- Right now cross attention model is using basic attention mechanism i.e. importance of test points for each train point (where query is train points and key and value are test points), but intuitively this does not sound right. Lets discuss this in next meeting