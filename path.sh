CONDA=/aifs/users/wd007/software/miniconda3
if [ -e $CONDA/etc/profile.d/conda.sh ]; then
    #source $CONDA/etc/profile.d/conda.sh && conda deactivate && conda activate py27tf12
    source $CONDA/etc/profile.d/conda.sh && conda deactivate && conda activate py27tf14hvd
else
    source $CONDA/bin/activate
fi


export OPENFST=/aifs/users/wd007/speaker/data/tools/fst/openfst
#export OPENFST=/aifs/users/wd007/speaker/data/tools/kaldi-debug/tools/openfst
export TENSORFLOW_SRC_PATH=/aifs/users/wd007/software/miniconda3/envs/py27tf14hvd/lib/python2.7/site-packages/tensorflow

CUDAROOT=/aifs/tools/CUDA/cuda-10.0
NCCL_ROOT=/aifs/tools/NCCL/nccl-10.0

#GCCROOT=/aifs/tools/software/gcc-4.9.2
#GCCROOT=/aifs/tools/software/gcc-5.4.0
#EXTLIB=/home/feipeng.pf/proj/tools/gcc/gmp-4.3.2/gmp-build/gmp-4.3.2/lib:/home/feipeng.pf/proj/tools/gcc/mpfr-2.4.2/mpfr-build/mpfr-2.4.2/lib:/home/feipeng.pf/proj/tools/gcc/mpc-0.8.1/mpc-build/mpc-0.8.1/lib

#export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$CUDAROOT/lib64:$EXTLIB:$GCCROOT/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$CUDAROOT/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$NCCL_ROOT/lib:$LIBRARY_PATH
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT
#export PATH=$CUDAROOT/bin:$GCCROOT/bin:$PATH
export PATH=$CUDAROOT/bin:$PATH
