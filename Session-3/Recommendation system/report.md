# Epoch Spring Camp 2026
## Take Home Assignment 3 Report

### 1. Objective
The goal of this assignment was to build a neural recommender system for implicit feedback data and compare a simple collaborative filtering baseline against a neural alternative. I implemented:

- Matrix Factorization (MF) as the baseline
- An MLP-based recommender
- A bonus hybrid model inspired by Neural Collaborative Filtering (NeuMF)

The main question was whether a neural interaction function can model user-item relationships better than a simple dot product, and whether combining both approaches improves performance further.

### 2. Dataset and Setup
The dataset contains only positive user-item interactions.

- Users: `942`
- Items: `1447`
- Interactions: `55,375`
- Columns: `user_id`, `item_id`

The interaction distribution is uneven:

- User interactions: mean `58.78`, median `39.5`, max `378`
- Item interactions: mean `38.27`, median `13`, max `501`

This confirms that the data is sparse and imbalanced, which is typical for recommendation problems.

I used a leave-one-out style split:

- Test set: one interaction per user, `942` rows
- Validation set: one interaction per user from the remaining data, `942` rows
- Training set: `53,491` rows

Since the data is implicit, only positive interactions are observed. To create a binary learning task, I sampled negative items for each user. In the training loader, I used `10` negatives per positive. I also added a user-dependent weight `1 / (count + alpha)` with `alpha=10` so that highly active users do not dominate the loss.

### 3. Models Implemented
#### 3.1 Matrix Factorization
For the baseline, each user and each item is represented by an embedding vector. The relevance score is the dot product of the two embeddings:

`score(u, i) = <p_u, q_i>`

This is a simple and efficient collaborative filtering model, but it only captures linear interactions in the latent space.

I used:

- User embedding + item embedding
- Embedding dimension: `16`
- Loss: weighted `BCEWithLogitsLoss`
- Optimizer: Adam
- Early stopping on validation Hit@10

#### 3.2 MLP-based Recommender
To make the interaction function more expressive, I replaced the dot product with a neural network. User and item embeddings are concatenated and passed through an MLP:

- Embedding dimension: `32`
- MLP: `Linear(64 -> 64) -> ReLU -> Linear(64 -> 32) -> ReLU -> Linear(32 -> 1)`

This lets the model learn nonlinear combinations of user and item features instead of relying only on similarity in embedding space.

#### 3.3 Bonus: NeuMF-style Hybrid
For the bonus task, I combined both approaches in a hybrid model:

- MF branch: element-wise product of user and item embeddings
- MLP branch: concatenated embeddings passed through hidden layers
- Final layer: concatenation of the MF and MLP outputs, followed by a linear prediction layer

I also warm-started the hybrid model using the pretrained MF and MLP embeddings saved earlier in the notebook.

### 4. Training and Evaluation
All models were trained in PyTorch with Adam and early stopping. The main evaluation metric was `Hit@K`, which checks whether the held-out positive item appears in the top `K` recommendations among sampled negatives.

I also evaluated metric sensitivity by varying:

- `K` in `{5, 10, 20}`
- Number of sampled negatives in `{50, 100, 200, 500}`

One important observation is that evaluation uses randomly sampled negatives. Because of this, a single `Hit@10` estimate can vary from run to run. For the final fair comparison, I reset `np.random.seed(seed)` before evaluating each model in the completed bonus section.

### 5. Results
#### 5.1 Validation Behavior
The MF model improved steadily during training and early-stopped at epoch `28`, reaching validation `Hit@10` around `0.5180`.

The MLP model started much stronger and early-stopped at epoch `22`, reaching validation `Hit@10` around `0.6083`.

This already suggested that the neural interaction function was learning useful patterns that MF could not capture as easily.

#### 5.2 Test Results
Using the notebook evaluation at `K=10` and `100` sampled negatives:

- MF sensitivity grid gave `Hit@10 = 0.5425`
- MLP sensitivity grid gave `Hit@10 = 0.6210`

For the final controlled comparison in the completed hybrid section, I evaluated all three models with the same random seed before each test run:

| Model | Hit@10 |
|---|---:|
| MF | 0.5372 |
| MLP | 0.6189 |
| NeuMF | 0.6688 |

These results show two clear trends:

1. The MLP model outperformed the MF baseline.
2. The combined NeuMF model performed best, improving over both standalone approaches.

So the answer to the bonus question is yes: combining MF and MLP improved performance over using MF or MLP alone.

### 6. Additional Observations
#### 6.1 Sensitivity to Evaluation Settings
As expected, Hit@K decreases when the number of negative samples increases, because the ranking task becomes harder. For example:

- MF: `Hit@10` dropped from `0.7282` with `50` negatives to `0.2304` with `500` negatives
- MLP: `Hit@10` dropped from `0.7792` with `50` negatives to `0.3025` with `500` negatives

The MLP stayed consistently ahead of MF across the same evaluation settings.

#### 6.2 Embedding Visualization
I also tried visualizing the learned MF embeddings with PCA. The first two principal components explained only about:

- User embeddings: `15.87%`
- Item embeddings: `14.88%`

This suggests the learned structure is distributed across many dimensions, so 2D PCA is not sufficient to explain much of the latent geometry.

#### 6.3 Overfitting and Regularization
The hybrid model is more expressive, so it also carries more risk of overfitting. In this notebook, I controlled that mainly through:

- Early stopping
- Validation monitoring
- Lower learning rate for NeuMF (`5e-4`)
- Warm-starting from pretrained MF and MLP embeddings

### 7. Conclusion
This assignment showed the progression from classical collaborative filtering to neural recommendation.

- MF was a strong and simple baseline.
- The MLP model improved performance by learning nonlinear user-item interactions.
- The hybrid NeuMF model gave the best result, showing that linear and nonlinear interaction signals are complementary.

Overall, the notebook supports the idea behind Neural Collaborative Filtering: a dot-product model is useful, but combining it with a neural interaction function leads to better recommendation quality on this dataset.
