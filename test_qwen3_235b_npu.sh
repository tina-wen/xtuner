set -x

CANN_DIR=/usr/local/Ascend/ascend-toolkit # 默认CANN安装地址
CANN_OPS_DIR=/tmp/cann-ops # GMM_NZ使能补丁的CANN-ops安装地址
PTA_FSDP_DIR=/usr/local/lib/python3.11/site-packages/torch_npu/distributed/fsdp # PTA的FSDP补丁位置，方便后续替换slice算子的实现

mkdir ${CANN_OPS_DIR}
./pta_patch/CANN-custom_ops--linux.aarch64.run --install-path=${CANN_OPS_DIR}
source ${CANN_OPS_DIR}/vendors/customize/bin/set_env.bash
source ${CANN_DIR}/set_env.sh

# 安装PTA 2.6.0版本GMM 切K轴补丁
pip install /path/to/torch_npu-custom.whl --force-reinstall
cp ./pta_patch/_fsdp_collectives.py ${PTA_FSDP_DIR}

# 使能GMM NZ开关
export GROUPMM_NZ_TRANSPOSE=1

export QWEN3_MOE_PATH=/path/to/qwen3_moe_weights
export ALPACA_PATH=/path/to/alpaca_dataset

export XTUNER_USE_FA3="1" 
export HCCL_RDMA_TC=132


# 自定义, 1是开启，0是关闭
export LINEAR_ONLY_SHARD=1

mkdir ${LOGS_DIR}

torchrun --nproc-per-node 16 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    ci/scripts/test_sft_trainer_235B.py \
    ${PROF_DIR} | tee ${LOGS_DIR}/rank_${RANK}.log
