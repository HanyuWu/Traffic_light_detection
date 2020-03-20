# Traffic light detection 

<img src = 'examples/Screenshot from 2020-03-20 00-45-29.png'><br>

* All you need is to open **CarND-Object-Detection-Lab.ipynb** in the jupyter notebook, so be sure you install the jupyter notebook in your python environment.<br>

    Documentation on working with [Anaconda environments](https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html#managing-environments).

* Be sure to install all the libraries presented in the **environment.yml**. You can manually setup your own environment in conda or you can try running the following command.<br>

```bash
conda env create -f environment.yml
conda activate carnd-advdl-odlab
```

**Be aware that you should install the right tensorflow version. If you don't have a GPU, then you should install the CPU version.**<br>

* **CarND-Object-Detection-Lab.ipynb** is well commented, please follow the procedure inside it. If it can run without error, then you can see how well the **ssd_mobilenet_v1** perform on the images which contain traffic light object in it.
