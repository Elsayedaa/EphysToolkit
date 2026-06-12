# EphysToolkit
Custom Python modules for handling electrophysiology data produced by the Cang Lab at UVA.

# Installation
<code>pip install https://github.com/Elsayedaa/EphysToolkit/archive/refs/heads/main.zip</code>

# Documentation

[EphysToolkit Documentation](https://elsayedaa.github.io/EphysToolkit/ephystoolkit/EphysToolkit.html)

# To access the demo

### 1) create anaconda environment 

- If you do not have anaconda installed, [click here](https://www.anaconda.com/download/success?reg=auth) to download.
- Run anaconda prompt and paste the following commands

```
conda create -n ephystoolkit python=3.9
```
```
conda activate ephystoolkit
```
### 2) Install Jupyter Notebook and register the environment   

```
pip install jupyterlab
```
```
python -m ipykernel install --user --name ephystoolkit --display-name "Python (ephystoolkit)"
```
### 3) Install EphysToolkit and other dependencies 

```
pip install https://github.com/Elsayedaa/EphysToolkit/archive/refs/heads/main.zip  
```
```
pip install matplotlib seaborn
```
### 4) Download the demo notebook and sample data (932 mb)

- Make sure to download these to the same folder:
- [Demo notebook (GitHub link)](https://github.com/Elsayedaa/EphysToolkit/blob/c767612a5182c4199746a470ce156e5c7d60051e/how_to_demonstration.ipynb) 
- [Dataset (Figshare link)](https://doi.org/10.6084/m9.figshare.32664219)
  
### 5) Launch the demo
- Go to the folder where the notebook and the dataset were downloaded
- Run an anaconda prompt from the folder
- Run the command: 
```
jupyter notebook 
```
- Click the demo notebook
