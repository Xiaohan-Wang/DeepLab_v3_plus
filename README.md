# DeepLab_v3_plus
An implementation of DeepLab v3+, with the reference on https://github.com/jfzhang95/pytorch-deeplab-xception. 

This project is used to 
* make myself get familiar with the basic structure of DL projects;
* decouple most modules so that they can be easily reused;
* learn [MLflow](https://www.mlflow.org/docs/latest/index.html) to clearly track experiments parameters, results, etc.

TODO list
* [ ] make a seperate template project


### Organizing runs in experiments
```python
# create experiment (this creates a specific experiment folder under mlruns)
mlflow experiments create --experiment-name base_trainer
# run experiment (each run creates a seperate folder under the experiment folder above)
export MLFLOW_EXPERIMENT_NAME=base_trainer 
python train.py
```
