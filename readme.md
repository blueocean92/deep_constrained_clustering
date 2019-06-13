# Code for ECMLPKDD 2019 Paper: [A Framework for Deep Constrained Clustering - Algorithms and Advances](https://arxiv.org/abs/1901.10061)

## Installation

#### Step 1: Clone the Code from Github

```
git clone https://github.com/blueocean92/deep_constrained_clustering
cd deep_constrained_clustering
```




#### Step 2: Install Requirements

**Python**: see [`requirement.txt`](https://github.com/blueocean92/deep_constrained_clustering/blob/master/requirements.txt) for complete list of used packages. We recommend doing a clean installation of requirements using virtualenv:
```bash
conda create -n testenv python=3.6
source activate testenv
pip install -r requirements.txt 
```

If you dont want to do the above clean installation via virtualenv, you could also directly install the requirements through:
```bash
pip install -r requirements.txt --no-index
```

**PyTorch**: Note that you need [PyTorch](https://pytorch.org/). We used Version 1.0.0 If you use the above virtualenv, PyTorch will be automatically installed therein. 


## Running Constrained Clustering Experiments

While in `deep_constrained_clustering` folder:

#### Step 1: Download Pretrained Networks

```
sh download_model.sh
```

#### Step 2: Download Processed Reuters Data(optional, MNIST and Fashion is available in torchvision.datasets)

```
sh download_data.sh
```

```
cd experiments/
```

While in `deep_constrained_clustering/experiments` folder:
#### Step 3: Run Experimental Scripts to Reproduce Results

###### Option 1: Run Demo Pairwise Constraints Script

To run the pairwise constrained clustering using pre-trained weights (AE features, 6000 constraints), do:
```bash
python run_DCC_pairwise.py --data $DATA
```

For the `--data` flag which specifies the data set being used. The options are "MNIST", "Fashion" and "Reuters".

To run the pairwise constrained clustering from raw features, do:
```bash
python run_DCC_pairwise.py --data $DATA --use_pretrain 'False'
```

###### Option 2: Run Demo Instance Constraints Script

To run the instance difficulty constrained clustering, do:
```bash
python run_DCC_instance.py --data $DATA
```

###### Option 3: Run Demo Triplets Constraints Script

To run the triplets constrained clustering (6000 constraints), do:
```bash
python run_DCC_triplets.py --data $DATA
```


###### Option 4: Run Demo Global Constraints Script

To run the global size constrained clustering, do:
```bash
python run_DCC_global.py --data $DATA
```


###### Option 5: Run Demo Improved DEC Script

To run the baseline Improved DEC, do:
```bash
python run_improved_DEC.py --data $DATA
```



