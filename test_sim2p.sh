#!/bin/sh
MODEL_PATH=/path/to/the/model/checkpoint  # path to the trained model checkpoint
CHURN_STEP_RATIO=0.3
GUIDANCE=1
SPLIT=$6

source ./args.sh $NGPU $DATASET_NAME $TAR_MODA $PRED $UNET_TYPE $DIMS $ATTN $EXP $DIT_TYPE $BS $IMG_SIZE $NUM_CH $NUM_RES_BLOCKS $COND $SIGMA_DATA $SIGMA_MAX $SIGMA_MIN $COV_XY $USE_16FP $BETA_D $BETA_MIN $TAB_DIM $TAB_MODE

N=5
GEN_SAMPLER=heun
BS=16
NGPU=1
ATTN_TYPE=flash
USE_16FP=True

if [[ $TAR_MODA == 'PETSSP' ]]; then
      IN_CH=2
else
      IN_CH=1
fi

mpiexec --mca btl vader,self -n $NGPU python3 sim2p_test.py --dims=$DIMS --dit_type=$DIT_TYPE --tab_dim=$TAB_DIM  --tab_mode=$TAB_MODE \
 --exp=$EXP --save_syn_scans True --unet_type=$UNET_TYPE \
 --target_modality=$TAR_MODA \
 --in_channels $IN_CH \
 --batch_size $BS --churn_step_ratio $CHURN_STEP_RATIO --steps $N --sampler $GEN_SAMPLER \
 --model_path $MODEL_PATH --attention_resolutions $ATTN --pred_mode $PRED \
 ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"}  \
 ${COND:+ --condition_mode="${COND}"} --sigma_data $SIGMA_DATA --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN --cov_xy $COV_XY \
 --dropout 0.1 --image_size $IMG_SIZE --num_channels $NUM_CH --num_head_channels 64 --num_res_blocks $NUM_RES_BLOCKS \
 --resblock_updown True --use_fp16 $USE_16FP --attention_type $ATTN_TYPE --use_scale_shift_norm True \
 --weight_schedule bridge_karras --data_dir=$DATA_DIR \
 --dataset=$DATASET_NAME --rho 7 --upscale=False ${CH_MULT:+ --channel_mult="${CH_MULT}"} ${UNET:+ --unet_type="${UNET}"} ${SPLIT:+ --split="${SPLIT}"} ${GUIDANCE:+ --guidance="${GUIDANCE}"}
