#!/bin/sh
DATASET_NAME=mri2pet
# CKPT='/path/to/checkpoint'  # path to checkpoint to resume training

source args.sh $NGPU $DATASET_NAME $SSP_PATH $TAR_MODA $PRED $UNET_TYPE $LR $DIMS $ATTN $EXP $DIT_TYPE $BS $IMG_SIZE $NUM_CH $NUM_RES_BLOCKS $COND $SAMPLER $SIGMA_DATA $SIGMA_MAX $SIGMA_MIN $COV_XY $SAVE_ITER $USE_16FP $BETA_D $BETA_MIN $DATASET $TAB_DIM $TAB_MODE

FREQ_SAVE_ITER=10000
NGPU=1
ATTN_TYPE=flash
USE_16FP=True

if [[ $TAR_MODA == 'PETSSP' ]]; then
      IN_CH=2
else
      IN_CH=1
fi

mpiexec --mca btl vader,self -n $NGPU python3 sim2p_train.py --dims=$DIMS --exp=$EXP --debug=True --unet_type=$UNET_TYPE \
 --attention_resolutions $ATTN --use_scale_shift_norm True  --dit_type=$DIT_TYPE --tab_dim=$TAB_DIM --tab_mode=$TAB_MODE \
 --in_channels $IN_CH \
 --ssp_projector_path=$SSP_PATH --target_modality=$TAR_MODA \
 --dropout 0.1 --ema_rate 0.9999 --batch_size $BS \
 --image_size $IMG_SIZE --lr $LR --num_channels $NUM_CH --num_head_channels 64 \
 --num_res_blocks $NUM_RES_BLOCKS --resblock_updown True ${COND:+ --condition_mode="${COND}"} ${MICRO:+ --microbatch="${MICRO}"} \
 --pred_mode=$PRED  --schedule_sampler $SAMPLER ${UNET:+ --unet_type="${UNET}"} \
 --use_fp16 $USE_16FP --attention_type $ATTN_TYPE --weight_decay 0.0 --weight_schedule bridge_karras \
  ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"}  \
 --data_dir=$DATA_DIR --dataset=$DATASET ${CH_MULT:+ --channel_mult="${CH_MULT}"} \
 --num_workers=8  --sigma_data $SIGMA_DATA --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN --cov_xy $COV_XY \
 --save_interval_for_preemption=$FREQ_SAVE_ITER --save_interval=$SAVE_ITER ${CKPT:+ --resume_checkpoint="${CKPT}"} 
