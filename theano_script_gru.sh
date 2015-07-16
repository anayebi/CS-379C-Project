CUDA_ROOT=/usr/local/cuda
export PATH=$PATH:/usr/local/cuda/bin
THEANO_FLAGS=device=gpu0 python gru_train_vsoma.py
