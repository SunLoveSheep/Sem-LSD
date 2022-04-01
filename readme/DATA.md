# Dataset preparation

To train the model, data annotations should be converted to COCO format and arranged in the following format:

  ~~~
  ${DATA_PATH}
  |-- annotations
      |-- train_semantic_line.json
      |-- val_semantic_line.json
  |-- images
      |-- train_semantic_line
      |-- train_semantic_line_mag
      |-- val_semantic_line
      |-- val_semantic_line_mag 
  ~~~

where train_semantic_line / val_semantic_line folders contain the images used for training and validating.

[Optional] train_semantic_line_mag / val_semantic_line_mag folders contain the filtered gradient magnitude images processed by 
selected edge operators (such as Scharr or Sobel) used for Magnitude Loss.

Currently we have 5 KAIST_URBAN sequences sampled (1 selected from every other 5 images) and labeled:

| Data set | # Labeled  |
|---|---|
| KAIST seq26  | 1946  |
| KAIST seq29  | 1479  |
| KAIST seq30  | 4227  |
| KAIST seq38  | 4318  |
| KAIST seq39  | 3729  |
| KITTI | 14999 |

Splited train/val images and annotations in COCO format as well as .xml format can be found:

KAIST URBAN:
[Labels](https://drive.google.com/file/d/1DviC3huFzjDh42jkwwMAkN7k-ZvEa4rG/view?usp=sharing)

As for the images, please refer to the official website. We use the stereo_left images.

KITTI:
[Images](https://drive.google.com/file/d/1O1QU46uNawF8zmVzWRXYLE3EESM4kkTK/view?usp=sharing) / 
[Labels](https://drive.google.com/file/d/1ARfByYB5EHsLpI4ChV-x26VnMpK4WR7r/view?usp=sharing)

## To generate image gradient magnitudes

Check ```src/_get_gradient_magnitude_images.py```