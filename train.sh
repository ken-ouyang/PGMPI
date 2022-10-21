TB_SAVEIMAGE=5
MASK_WEIGHT=10
LLFF_WIDTH=180
OFFSET=90
SUBLAYERS=12
LAYERS=16


SCENE=sample_scene
MODEL_DIR=debug
RUNPATH=./runs/
EVAL_PATH=./runs/evaluation/

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -no_webgl -scene $SCENE -model_dir $MODEL_DIR  -sublayers $SUBLAYERS -layers $LAYERS -llff_width $LLFF_WIDTH -tb_saveimage $TB_SAVEIMAGE -lrc 0.1 -sigmoid_offset 5 -offset $OFFSET\
                         -gpus 4 -ref_img images/000005_video_0.png -runpath $RUNPATH -epochs 500 -mask_weight $MASK_WEIGHT -use_appearance -use_learnable_planes -gradloss 0.05 -perceptualloss 0.01 -sparsityloss 0 
