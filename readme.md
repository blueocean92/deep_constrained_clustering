## Installation

### Step 1: Clone the Code from Github

```
git clone https://github.com/blueocean92/deep_constrained_clustering
cd deep_constrained_clustering/experiments
```




### Step 2: Install Requirements

**Python**: see [`requirement.txt`](https://github.com/StanfordVL/taskonomy/blob/master/taskbank/requirement.txt) for complete list of used packages. We recommend doing a clean installation of requirements using virtualenv:
```bash
conda create -n testenv python=3.6
source activate testenv
pip install -r requirement.txt 
```

If you dont want to do the above clean installation via virtualenv, you could also directly install the requirements through:
```bash
pip install -r requirement.txt --no-index
```

**PyTorch**: Note that you need [PyTorch](https://pytorch.org/). We used Version 0.4.1 If you use the above virtualenv, PyTorch will be automatically installed therein. 


## Running Constrained Clustering Experiments

While in `deep_constrained_clustering/experiments` folder:

#### Option 1: Run Demo Pairwise Constraints Script

To run the pairwise constrained clustering using pre-trained weights(AE features), do:
```bash
python run_DCC_pairwise.py --data $DATA
```

For the `--data` flag which specifies the data set being used. The options are "MNIST", "Fashion" and "Reuters".

To run the pairwise constrained clustering from raw features(end-to-end learning), do:
```bash
python run_DCC_pairwise.py --data $DATA --use_pretrain False
```

#### Option 2: Run Demo Instance Constraints Script

To run the instance difficulty constrained clustering, do:
```bash
python run_DCC_instance.py --data $DATA
```

For the `--data` flag which specifies the data set being used. The options are "MNIST", "Fashion" and "Reuters".

#### Option 3: Run Demo Triplets Constraints Script

To run the triplets constrained clustering, do:
```bash
python run_DCC_triplets.py --data $DATA
```

For the `--data` flag which specifies the data set being used. The options are "MNIST", "Fashion".

#### Option 4: Run Demo Global Constraints Script

To run the global size constrained clustering, do:
```bash
python run_DCC_global.py --data $DATA
```

For the `--data` flag which specifies the data set being used. The options are "MNIST", "Fashion".

#### Option 5: Run Demo Improved DEC Script

To run the baseline Improved DEC, do:
```bash
python run_improved_DEC.py --data $DATA
```

For the `--data` flag which specifies the data set being used. The options are "MNIST", "Fashion" and "Reuters".


