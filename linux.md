# Jupyternotebook on server
```
 python3 -m venv myenv
 python3 -m venv myenv
pip install ipykernel jupyter
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
jupyter notebook --no-browser --port=8888
```
