When using this system, place the programs from the corresponding subfolders within the ‘mmclassification’ and ‘mmediting’ folders into their respective folders to form a new neural network model.

The cases and preprocessing programs for simulated and real-world data can be found in the 'example_image' folder, as well as the beads counting program.

In the folder 'newdataset', the program generate_psf_aug.py can be used to generate images after Richardson-Lucy Deconvolution. The at-focus PSF kernel can be read from test_psf_ori1.txt.

By using the rest of the programs, the edge-gradient and part-sizes can be read from images, and the defocus parameters can be read from the json files by read_json_defocus.py.

requirements:

mmediting==0.16.0 (or ≥ 0.16.0, ≤1.0.0)
mmclassification==0.23.2 (or ≥ 0.23.0, ≤1.0.0)
pytorch==1.10.0
opencv-contrib-python==4.5.4.60
grad-cam==1.3.9 (or ≥ 1.3.9)




