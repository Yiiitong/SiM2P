#!/bin/sh
BS=1 
LR=0.0001 
DIMS=3
DATASET_NAME='mri2pet'
PRED='vp'
UNET_TYPE='dit'  # 'adm' or 'dit' 
TAR_MODA='PET' #'PETSSP' or 'PET'
TAB_DIM=26
TAB_MODE='concat'  # 'concat' or 'add'
DIT_TYPE='DiT-XL/4' # DiT-XL/4 DiT-XL/8, DiT-L/4, DiT-B/4, DiT-S/4, ...
SSP_PATH='' # path to pretrained ssp projector, if available
NGPU=1
DEBUG=False

SIGMA_MAX=80.0
SIGMA_MIN=0.002
SIGMA_DATA=0.5
COV_XY=0

NUM_CH=64
ATTN=16,8,4,2,1
SAMPLER=real-uniform
NUM_RES_BLOCKS=2
USE_16FP=True
ATTN_TYPE=flash

if [[ $DATASET_NAME == "mri2pet" ]]; then
    DATASET=mri2pet
    IMG_SIZE=80
    NUM_CH=64

    if  [[ $UNET_TYPE == "dit" ]]; then
        EXP="mri2pet${IMG_SIZE}_${DIT_TYPE}_lr${LR}_${DIMS}D_MSE_Tab${TAB_MODE}_TabDim${TAB_DIM}"
    else
        NUM_RES_BLOCKS=2
        EXP="mri2pet${IMG_SIZE}_${UNET_TYPE}_resb${NUM_RES_BLOCKS}_ch${NUM_CH}_lr${LR}_${DIMS}D_MSE_Tab${TAB_MODE}"
    fi
     
    SAVE_ITER=1000
else
    echo "Not supported"
    exit 1
fi
    
if  [[ $PRED == "ve" ]]; then
    EXP+="_ve"
    COND=concat

elif  [[ $PRED == "vp" ]]; then
    EXP+="_vp"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
elif  [[ $PRED == "ve_simple" ]]; then
    EXP+="_ve_simple"
    COND=concat
elif  [[ $PRED == "vp_simple" ]]; then
    EXP+="_vp_simple"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
else
    echo "Not supported"
    exit 1
fi


if  [[ $TAR_MODA == "PETSSP" ]]; then
    EXP+="_petssp"
fi
