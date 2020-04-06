# OptiFlex

Video-Based animal pose estimation using deep learning enhanced by optical flow.

## Installation

***Python version >=3.6 required.***

***CUDA 10.0 highly recommended.***

#### Required Packages

- [Tensorflow](https://www.tensorflow.org/) `pip install tensorflow==1.12` or `pip install tensorflow-gpu==1.12` *(Basic requirements)*
- [Keras](https://keras.io/) `pip install keras` *(Basic requirements)*
- [NumPy](https://numpy.org) `pip install numpy` *(Basic requirements)*
- [SciPy](https://www.scipy.org) `pip install scipy` *(Basic requirements)*
- [OpenCV](https://opencv.org) `pip install opencv-python` *(Basic requirements)*
- [Matplotlib](https://matplotlib.org) `pip install matplotlib` *(Required for plotting and visualization only)*
- [scikit-image](https://scikit-image.org) `pip install scikit-image` *(Required for multi-view correction only)*
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) `pip install PyQt5` *(Required for using GUI)*

## Walkthrough

*For users want to customize the package or frequently use scripts, we highly recommend using a proper IDE.*

#### Video Acquisition

For raw videos, we have no hard requirements for format, general suggestions are as follows:

- Use uncompressed or lossless-compressed encoding for best quality, `*.avi` and `*.mp4` is suggested.
- Sampling rate of video can be freely adjusted with your requirement.
- Aspect ratio can be any, but avoid over-exaggerated ratio (e.g. 8:1).
- Colorspace of frames can be any, using normal Grayscale or RGB is suggested.
- Audio is not needed.

#### Data Annotation

This package use JSON (JavaScript Object Notation) as basic label file format. Format conversion is required if other label file type is used. The format definition as follows:

```
[
    {"left": %d, "top": %d, "width": %d, "height": %d, "label": %s},
    ... ...
]
```

> - "left" (int): defined as x-pixel location from LEFT side of frame.
> - "top" (int): defined as y-pixel location from TOP side of frame.
> - "width" (int): defined as x-pixel size of the label.
> - "height" (int): defined as y-pixel size of the label.
> - "label" (str): defined as the name of the joint.

We have provided MATLAB® codes for converting label files from LocoMouse [*(Machado et.al. 2015)*](https://doi.org/10.7554/eLife.07892) and LEAP [*(Pereira et.al. 2019)*](https://doi.org/10.1038/s41592-018-0234-5). For other packages, user may need to customize codes for file type conversion.

We highly recommend using our GUI `Manual_Labelling_Tool.py` for data annotation.

1. Create a `*.csv` file containing label definition, each line defined as `label_name, width, height, hex_color_to_display`. An example is given in `/examples/label_list.csv`.
2. Load a video file, a label CSV, and define a output folder at the loader screen.
3. After successful loading, user could explore the annotation GUI under `Preview Mode`. When change to `Record Mode`, all labels and frames will be saved.
4. Basic operations as follows:
    1. User could use `A` and `D` key to navigate previous and next frame, `W` and `S` key to select between labels.
    2. User also able to select a label with `Left Mouse Click`, add label with `SHIFT + Left Mouse Click`, and delete label with `Right Mouse Click`.
    3. Moving a selected label with `Mouse Dragging`.
5. When annotation is finished, simply `Exit` the GUI as the saving is automatic under `Record Mode`.
6. Upon finishing the annotation, user could use the script `label_conversion_json2heatmap.py` in `/scripts` to convert label to HeatMap format.

#### Dataset File Structure

The only dataset file structure required for this package is putting frame/label files from the same video in one folder. But we do have some suggestions for the best practice of data arrangement.
A suggested dataset file structure is as follows:

```
major_data_folder/            <- Suggest to use animal or project name
    data_notes.txt            <- Optional
    minor_data_folder_1/      <- Suggest to use yyyymmdd_xxxx format, create one folder for each video
        data_notes.txt        <- Optional
        grp_frm_ddddd.png     <- All frame images. If video containing mutliview data, group name should be as prefix and followed by an underscore
        grp_frm_ddddd.json    <- All JSON labels, the name should be the same as frame images.
        grp_frm_ddddd.pkl     <- All HeatMap labels, the name should be the same as frame images.
    minor_data_folder_2/
        ... ...
    ... ...
```

#### Dataset Creation

1. To start creating datasets for training the network, you will need one `*.csv` file for each type (Training/Validation/Test) of dataset. Each line in the `*.csv` file defined as `minor_data_name, minor_data_path`. An example is given in `/examples/dataset_list.csv`.
2. After having the CSVs required, user could either use a GUI `Dataset_Creation_Tool.py` or scripts in `/scripts` to create required datasets.
    1. For users using the GUI
        - Select each `*.csv` file under Training/Validation/Test dataset. Please disable (select `No`) certain dataset if not needed.
        - Please input the name of labels **EXACTLY** as it appears in the label files, any miss-input will cause output error.
        - If user is using a multiview recording system, please input name of groups **EXACTLY** as it appears in the prefix of files.
        - Missing/Incomplete labels will still be saved if `Keep Labels with Dummy Data` is checked. *(Not recommended)*
        - All data in the [Test Set] will be saved if `Always Save Files form Test Sets` is checked. *(Recommended)*
        - `Augmentation Number` can be arbitrary number, but too large number of augmentation would cause high homogeneity in dataset, thus leads to overfitting.
        - `Process Core` should be set based on user machine spec.
        - If the dataset will be used for OpticalFlow model, please **ALWAYS** check `Sequential` under `HM Random`.
        - For other session not mentioned here, user could adjust as their own will.
    2. For users using scripts
        - All variables are identical to GUI, and have been clearly documented inside the scripts.
        - If user wish to generate a dataset for JSON label, please use `create_json_data_sets.py` in `/scripts`.
        - If user wish to generate a dataset for HeatMap label only to use with FlexibleBaseline, please use `create_heatmap_data_sets.py` in `/scripts`.
        - If user wish to generate a dataset for HeatMap label also to use with OpticalFlow, please use `create_heatmap_sequential_data_sets.py` in `/scripts`.
    3. For all users
       - The image and label **MUST** be resized to a pixel size of *power of 2* (e.g. 128, 256) for both width and height. Please choose a proper combination most close to the original aspect ratio of video.
       - For the peak value of HeatMap, *16* is highly suggested. User also could define a peak value by their own, but a higher value usually leading to a better result.

#### Base Model Training and Inference

Before starting the model training process, it is crucial to have a proper folder structure setup for the code to properly access dataset and store model files. Thus, we highly recommend the following folder structure:
```
project_directory/            
    dataset/                  <- folder for storing all the annotated datasets
        trn_dataset1/     
        ver_dataset1/
        tst_dataset1/
        trn_dataset2/      
        ... ...
    model/                    <- folder for storing model and training history files
    OptiFlex/                 <- folder for code cloned from Github
```
1. Once users have created annotated dataset(s) and setup the proper folder structure,  user need to specify the absolute paths of required folder locations in `hm_train_script.py`, along with the basic dataset and model information.
2. Next, the user need to indicate the 3 basic training hyperparameters values:
    - `epochs`: number of times to go through the train set during training
    - `batch_size`: number of frames to use in a single training step (for gradient calculation)
    - `learning_rate`: magnitude of gradient update on the model weights
3. With all the parameters specified, the user can simply run `hm_train_script.py` to train the model and generate inference on the test set.    
4. After each epoch of training, the model will run inference on the verification set to produce some intermediate evaluation results, in terms of mean square error and mean absolute error. User can use these results to decide if the training is going as expected or experiencing overfitting/underfitting.
5. When the model is trained, the model files and training history files will be saved in the `model` folder.
6. Inference on the test set is automatically executed at the conclusion of the training process. The resulting heatmap, JSON, and labelled images will be stored in seperate folders in the `dataset` folder.  

#### OpticalFlow Training and Inference
OpticalFlow training process requires preprocessed tensors for model training
1. User must first obtain base model inference results on train/verification/test sets, which can be done by running `create_baseline_inferene_files.py` with appropriate folder paths specified.
2. Next, user will obtain the optical flow warped range tensors of train/verification/test sets, ready for training, with `opticalflow_precalculate.py` using the base model inference results from previous step.
3. Finally, running `hm_train_script.py` will execute the training and inference process, very similar to the base model training process.
4. The trained OpticalFlow model will be saved in the `model` folder and inference results will be saved in separate folder in `dataset` folder.


## Contents

- `OptiFlex.data` Package for data processing
    - `lbl_proc.py` Label I/O, conversion and visualization functions
    - `img_proc.py` Image & image-label enhancement and transform functions
    - `vid_proc.py` Video extraction and generation functions
    - `post_pred.py` Post-prediction visualization and processing functions
    - `locomouse_func.py` Functions exclusively for processing LocoMouse data
    - `dataset_func.py` Data wrangling functions for training preprocessing
    
- `OptiFlex.training` Package for training functions and scripts
    - `train_func.py` Data generators and model training functions
    - `post_train.py` Post processing and inference functions
    - `hm_train_script.py` Training script for heatmap generating base models
    - `flow_train_script.py` Training script for optical flow models
    
- `OptiFlex.utils` Package containing tools and utilities
    - `base_func.py` Basic functions required by various codes in this project
    - `disp_func.py` Data visualization functions
    - `eval_func.py` Model prediction results evaluation functions
    - `/matlab` MATLAB® codes for data conversion *(this folder is excluded from Python package)*
        - `locomouse_lbl_conv.m` Convert LocoMouse label files to JSON label files
        - `BAT_locomouse_lbl_conv.m` Batch processing version of `locomouse_lbl_conv.m`
        - `leap_h5_conv.m` Extract image files form LEAP `*.h5` files
        - `leap_data_conv.m` Convert data form LEAP dataset to JSON-PNG files
        
- `OptiFlex.gui` Package containing GUI related codes
    - `*.ui` PyQt5 GUI raw designing files
    - `manlbl_gui_desg.py` Manual data labelling GUI designing codes
    - `manlbl_gui_func.py` Manual data labelling GUI main functions
    - `setcrt_gui_desg.py` Dataset creation GUI designing codes
    - `setcrt_gui_func.py` Dataset creation GUI main functions
    - `setcrt_gui_scpt.py` Dataset creation GUI processing script
    
- `OptiFlex/scripts` Various Python scripts for data processing
    - Scripts are intuitively named and with detailed documentation inside codes.
    
- `OptiFlex/examples` Folder storing various example files for reference *(this folder is excluded from Python package)*
- `Manual_Labelling_Tool.py` GUI caller for manual labelling
- `Dataset_Creation_Tool.py` GUI caller for dataset generation and augmentation
