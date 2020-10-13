# SD-GAR
This repository hosts the experimental code for NeurIPS 2020 paper "Sampling-Decomposable Generative Adversarial Recommender".

## Requirements

The code is based on Tensorflow-2.1.0.

## Input Format

Each dataset is split into two files as the training set and testing set.

For each line in the files, it is consist of three elements split by the delimiter '\t'':

u   i   score

presenting the user 'u' rates the item 'i' a score of 'score'

Note that in our code, the sample is regarded as positive when the score is not higher than 4. 
Anyone who wants to change the threshold can modify it in load_data function in utils.py.

## Training the model

### config
To train a SD-GAR model, please refer to the config.py first. It contains the hyperparameters of the model. 
For example, for the citeulike dataset:
```
conf['citeulike'] = {
    "output_dir": "results/citeulike",
    "train_file": "data/citeulike/train.txt",
    "test_file": "data/citeulike/test.txt",
    "CUDA_VISIBLE_DEVICES": "3",
    "batch_size": 512,
    "learning_rate": 0.001,
    "dis_lambda": 0.05,
    "dis_sample_num": 5,
    "dis_emb_dim": 32,
    "gen_sample_num_item": 64,
    "gen_sample_num_user": 64,
    "gen_emb_dim": 64,
    "T": 0.5,
    "lambda_x": 0.1,
    "lambda_y": 1.,
}
```

Prefix 'dis' denotes the corresponding hyperparameter belong to the discriminator. 
Specifically, 'dis_lambda' denotes the coefficient of L2 regularization.
'dis_sample_num' denotes the number of negative samples when optimizing the discriminator. 
'dis_emb_dim' denotes the dimension of the user/item embeddings.

Prefix 'gen' denotes the corresponding  hyperparameter belong to the generator.
'gen_sample_num_item' and 'gen_sample_num_user' denote the number of samples when optimizing the user and item embeddings in the generator.
In our experiments, they are always set as the same value.
'gen_emb_dim' denotes the dimension of the user/item embeddings.

'T', 'lambda_x' and 'lambda_y' are hyperparameters of temperature. Specifically, 'T' is the temperature in the objective function in the discriminator 
while 'lambda_x' and 'lambda_y' are the temperature when optimizing the generator.

### training
After set up the config, run:
```angular2
    python main.py citeulike
```
