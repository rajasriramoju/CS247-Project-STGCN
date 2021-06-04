# CS247-Project-STGCN

## Execution instructions

### STGCN 

```
   cd STGCN
   python3 main.py
```
To change the epoch values, you can modify line 31 of `main.py`.  

### STSGCN

Implementation: https://github.com/Davidham3/STSGCN
How to run the code with our dataset:
Step 1: Install `docker` and `nvidia-docker`
Step 2: Clone the repo and build
```
$ git clone https://github.com/Davidham3/STSGCN
$ cd docker && docker build -t stsgcn/mxnet_1.41_cu100 .
```
Step 3: After building, move the data into corresponding folders:  
```
$ cp -r CS247-Project-STSGCN/STSGCN/data/PEMS07-12 STSGCN/data/
$ cp -r CS247-Project-STSGCN/STSGCN/config/PEMS07-12 STSGCN/config
```

The structure should be like this:  
```
├── config
│   ├── PEMS03
│   │   ├── individual_GLU_mask_emb.json
│   │   ├── individual_GLU_nomask_emb.json
│   │   ├── individual_GLU_nomask_noemb.json
│   │   ├── individual_relu_nomask_noemb.json
│   │   └── sharing_relu_nomask_noemb.json
│   ├── PEMS04
│   │   ├── individual_GLU.json
│   │   ├── individual_GLU_mask_emb.json
│   │   ├── individual_relu.json
│   │   ├── sharing_GLU.json
│   │   └── sharing_relu.json
│   ├── PEMS07
│   │   └── individual_GLU_mask_emb.json
│   ├── PEMS07-12
│   │   ├── individual_GLU_mask_emb.json
│   │   ├── individual_GLU_nomask_emb.json
│   │   ├── individual_GLU_nomask_noemb.json
│   │   ├── individual_relu_nomask_noemb.json
│   │   └── sharing_relu_nomask_noemb.json
│   └── PEMS08
│       └── individual_GLU_mask_emb.json
├── data
│   └── PEMS07-12
│       ├── convert_data.py
│       ├── PEMS07-12.csv
│       ├── PEMS07-12.npz
│       └── PEMS07-12.txt

```
Step 4: Run the model with the command below
```
$ docker run -ti --rm --runtime=nvidia -v $PWD:/mxnet stsgcn/mxnet_1.41_cu100 python3 main.py --config config/PEMS07-12/individual_GLU_mask_emb.json
```

Note: To vary the model configuration, you can change the config files in the command.  

Some side-note: I highly discourage anyone trying to install `nvidia-docker` on Windows WSL at the time of writing (June 03 2021). It was an absolute nightmare for me.  

### GMAN 

```
  cd GMAN/PeMs/
  python3 train.py
  python3 test.py
```


## Dataset downlooad

The data files required for this project can be downloaded from:

1. STGCN data from: https://drive.google.com/drive/folders/1LOMXNq6M6hWasKnog8-c6ttDQN4RLXSF?usp=sharing  
   This needs to be downloaded into the home directory
3. GMAN data from: https://drive.google.com/drive/folders/1N4xfWaUvqy2mPEPK8_OpIt9l7wCIINLx?usp=sharing  
   This needs to be downloaded into CS247-Project-STGCN/GMAN/PeMs/data

## References
This repo contains code from:
https://github.com/zhengchuanpan/GMAN  
https://github.com/Davidham3/STSGCN
