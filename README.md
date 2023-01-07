# datethesong
### scope
datethesong project aims to help the user in choosing the best linear regression model among several algorithm candidates to predict the release year of a given song.

### Virtual environment
The present project make use of virtual environment (venv) method. To work on the corresponding venv, run the following command in the parent folder of `datethesong`:
```source bin/activate```

### Run specific files
To run specific files, please place yourself in the parent folder of `datethesong` file and use python3 command line. Example:
```python3 datethesong/algorithms/data_utils.py```

### Remarks
- Required packages are listed in the ```/requirements.txt``` file
- Most of the figures are obtained by running the ```datethesong/experiments/plot_exp_training_size.py``` file. All obtained figures are saved into the ```/figures``` folder.
- If you have a ```No module named ...``` error, please run the following command in the parent directory of the datethesong folder: ```export PYTHONPATH=.```. This sets your ```PYTHONPATH``` to ".", which basically means that your ```PYTHONPATH``` will now look for any called files within the directory you are currently in, (and more to the point, in the sub-directory branches of the directory you are in. So it doesn't just look in your current directory, but in all the directories that are in your current directory) [[ref]](https://stackoverflow.com/questions/338768/python-error-importerror-no-module-named).


