



# Environment setup

## Creating a new environment
Based on the provided environment.yml file, we can create a new environment with the following command:

```bash
    conda env create -f environment.yml
```
This will create a new environment called feature_importance.

## Registering the environment in Jupyter

```bash
python -m ipykernel install --user --name=feature_importance
```
