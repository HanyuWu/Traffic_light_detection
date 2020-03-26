# Traffic light detection 

<img src = 'examples/Screenshot from 2020-03-20 00-45-29.png'><br>
<img src = 'examples/Screenshot from 2020-03-25 21-17-07.png'><br>

* All you need is to open **CarND-Object-Detection-Lab.ipynb** in the jupyter notebook, so be sure you install the jupyter notebook in your python environment.<br>

    Documentation on working with [Anaconda environments](https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html#managing-environments).

* Be sure to install all the libraries presented in the **environment.yml**. You can manually setup your own environment in conda or you can try running the following command.<br>

```bash
conda env create -f environment.yml
conda activate carnd-advdl-odlab
```

**Be aware that you should install the right tensorflow version. If you don't have a GPU, then you should install the CPU version.**<br>

* **CarND-Object-Detection-Lab.ipynb** and **CarND-Object-Detection-Lab-Onsite-test.ipynb** are well commented, please follow the procedure inside it. If it can run without error, then you can see how well the **ssd_mobilenet_v1** perform on the images which contain traffic light object in it.

* We have two models respectively for the 2 different detection task, simulator and onsite images. The test images are in the **onsite_traffic_light_img** file and **sim-traffic_light_img** file. You can either use the images inside these files or your own images to test the detectors.

* The frozen graphs for detection are in **onsite_frozen_interface** (on-site) and **ssd_mobilenet_v1_coco_TF1.4.0** (simulator).
