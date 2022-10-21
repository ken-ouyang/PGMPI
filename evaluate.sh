TB_SAVEIMAGE=5
MASK_WEIGHT=10
LLFF_WIDTH=180
OFFSET=90
SUBLAYERS=12
LAYERS=16

SCENE-sample_scene
MODEL_DIR=debug
RUNPATH=./runs/
EVAL_PATH=./runs/evaluation/


# for evaluation
CUDA_VISIBLE_DEVICES=0 python train.py -scene $SCENE -model_dir $MODEL_DIR -sublayers $SUBLAYERS -layers $LAYERS -llff_width $LLFF_WIDTH -tb_saveimage $TB_SAVEIMAGE -lrc 0.1 -sigmoid_offset 5 -offset $OFFSET\
                        -eval_path $EVAL_PATH  -gpus 1 -ref_img images/000005_video_0.png -runpath $RUNPATH -epochs 1000 \
                        -mask_weight $MASK_WEIGHT -predict -no_webgl \
                        -use_appearance \
                        -use_learnable_planes \
                        # -render_fixed_view \
                        # -render_depth \
                        # -no_eval \
                        # -no_video \


                          





