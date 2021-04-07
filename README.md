# Challenge Template

This project is intended to be used with the Unearthed CLI tool and is built in such a way that the Unearthed training and scoring pipelines will function.

## Requirements

This challenge is targetting the AWS SageMaker Scikit Learn Framework. Submissions must be compatible with this framework.
Details of this framework is available at https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html

## Submission Pipeline

The submission pipeline makes use of AWS SageMaker which runs within the AWS SageMaker Scikit Learn Framework container. The local Docker based simulation environment makes use of the same AWS SageMaker container, however the generation of predictions is slightly different.

The following steps are performed in the online submission pipeline:

### Step 1 - Model Training

The first step is an AWS SageMaker Training job.

In this step the source code of the submission is submitted into a SageMaker Training job with the entry point being the `__main__` function in `train.py`. The Training job executes within a standardized SageMaker container and has the full public data file mapped into the `/opt/ml/input/data/training` directory of the container.

It is the responsibility of `train.py` to call any required preprocessing (via the hook to the `preprocess` function in `preprocess.py`) to split the public data into inputs and target variables and perform the actual training.

Whilst it is possible to perform hyperparameter tuning within `train.py` it is recommended that this is performed offline as the training time during submission is limited to 1 hour.

Finally, the `train.py` must save the model to a binary file, and define a `model_fn` function that can read the binary file at a later point. This template demonstrates this using Pickle. Extensions to Pickle that save state or otherwise manipulate the training, prediction, and scoring environments are not supported, as they are ultimately not available in the industry partner's environment.

### Step 2 - Preprocess

Once a model has been produced from the Training job, the pipeline simulates the industry partner's production environment in two parallel pipelines, "public" and "private".

It is important to understand that the simulation is done in a batch mode, whereas in a real production environment each new prediction needs to be produced from a small subset of new data. The production environment has no knowledge of the targets or of any input data from the future.

The first step in both the public and private pipeline is a SageMaker Processing job which executes within the same standardized SageMaker container as the Training job. This job however calls the `__main__` function in the `preprocess.py` module and only the `preprocess.py` source code is available to this job.

The Processing job has the full public data file mapped into the for the public pipeline, however only has the input variables available for the private pipeline. Trying to perform processing on the target variables in the private pipeline may cause the submission to fail.

### Step 3 - Predictions

Once preprocessing steps have been applied to the public and private datasets, the pipeline simulates a prediction. To do this it runs a SageMaker Batch Transform job. In this step SageMaker deploys a standarized SageMaker container that contains model serving code. The Batch Transform job then orchestrates sending the results of the previous preprocessing step via a HTTP POST to a web endpoint hosted by the model serving container.

The model serving container makes several calls to functions in the `train.py` module of each submission.

The `model_fn` function is used to load the trained model into the container. The loaded model must be compatible with the AWS SageMaker Scikit Learn Framework. This function needs to load whatever format was used in the `save_model` function, which uses Pickle as a model serialization tool in this template. It is a good idea to leave this function as is.

The `input_fn` function is called when the Batch Transform job makes the HTTP call to the model serving container. The Batch Transform job sends the result of the preprocess step as a CSV file to the model serving container's web endpoint, which in turn passes it to this function. The output of this function must match the inputs expected by the previously trained model.

### Step 4 - Scoring

Once the prediction steps have been completed, the pipeline generates a score for the predictions made by the trained model. The pipeline does this by running a SageMaker Processing job that calls the `score` function in the `score.py` module. `score.py` is provided in the template, however it is not uploaded as part of a submission, and the latest version from the challenge template is always used.

It is important to note that the scoring function will compare predictions against the target variables in their untransformed, original format.
