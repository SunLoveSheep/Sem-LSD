cd src
python main.py ctdet_line \
--dataset semantic_line_kaist \
--skip3 \
--encode_type wh_direct \
--direct_loss cls \
--separate_wh 0 \
--wh_weight 1.0 \
--arch dla_34 \
--num_classes 14 \
--exp_id kaist_seq39_0816_dla34_COCOpret_NormGtSmGau_noInt_RndCrop_pytorch120 \
--batch_size 16 \
--lr 5e-4 \
--num_epochs 200 \
--lr_step 90,150 \
--gpus 0 \
--data_dir /workspace/tangyang.sy/Robotics_SemanticLines/data/KAIST_seq39_0816 \
--load_model ../models/ctdet_coco_dla_2x.pth \
#--load_model /workspace/tangyang.sy/pytorch_CV/pytorch_CenterNet/exp/ctdet_line/kitti_cont_13seqs_dla34_COCOpret_corrPolygon_noNormGtHM_debug/model_best.pth \
#--debug 2 \
#--center_thresh 0.25 \
#--not_rand_crop \
