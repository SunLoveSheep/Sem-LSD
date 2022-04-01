# RUNNING

## For training

A typical training script (check train_ctdet-line_kaist.sh):
~~~
python main.py 
ctdet_line \
--dataset semantic_line_kaist \
--num_classes 14 \
--direct_loss cls \
--input_res 512 \
--exp_id kaist_seq39_resnet18 \
--batch_size 16 \ 
--lr 5e-4 \
--num_epochs 300 \
--lr_step 90,150,220 \
--arch resnet_18 \
--data_dir /workspace/tangyang.sy/Robotics_SemanticLines/data/KAIST_seq39_0903 \
--load_model ../models/ctdet_coco_resdcn18.pth \
~~~

Meaning of each argument can be found in opt.py.

To include the gradient magnitude loss, add the following two lines:
~~~
--loss_hm_magnitude \
--hm_magnitude_weight 1.0 \
~~~
where loss type supports loss_hm_magnitude. If want to use positive only or negative only magnitude loss,
add also the following line like:
~~~
--loss_hm_magnitude \
--loss_hm_magnitude_neg_only \  # or _pos_only
--hm_magnitude_weight 1.0 \
~~~

hm_magnitude_weight is the weight factor, default is 1.0


## For inference
A typical inference script (check test_ctdetline_kaist.sh):
~~~
IMG_PTH=/workspace/tangyang.sy/pytorch_CV/test_imgs/KAIST_5seqs_20200214/Images
VIS_THRESH=0.25
EP=best
MODEL=kaist_5seqs_ResNet18_20200224
cd src
python demo.py ctdet_line \
--dataset semantic_line_kaist \
--num_classes 14 \
--demo ${IMG_PTH}/ \
--direct_loss cls \
--input_res 512 \
--arch dla_34 \
--load_model ../exp/ctdet_line/${MODEL}/model_${EP}.pth \
--save_path ${IMG_PTH}_${MODEL}_ep${EP}_vis${VIS_THRESH}/ \
--vis_thresh ${VIS_THRESH} \
--gpus 0 \
--save_img 0 \
#--loss_hm_magnitude \
cd ..
~~~


## For evaluation
For evaluation of the detection results, first make sure you have the detection results and 
the ground truth saved in .xml format, which is the default output format from current inference code.

Then set the ```root```, ```EXP_ID```, as well as ```SRC_GT``` accordingly in ```_analyze_semantic_line.py```
and run it.
~~~
python _analyze_semantic_line.py
~~~


# Pretrained models

We provide pretrained Sem-LSD models on KAIST_URBAN dataset.

arch: resnet_18  [Download model](https://drive.google.com/file/d/1aETg118596U0fdZtL1rHKSw6YlLQgOGj/view?usp=sharing)

arch: dla_34  [Download model](https://drive.google.com/file/d/1gT_vhZsl_LDXqX3djwwDPHfUuB2W5BjL/view?usp=sharing)