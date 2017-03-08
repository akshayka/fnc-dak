## NN Architecture
1. cmd-line flag for max_headline_len / max_body_len
2. Experiment: add layers, non-linearities

## Features
1. Remove stop-words, weight word vectors by TF-IDF
2. Experiment: Larger dimensional word vectors.

## Training word vectors
1. Experiment: do not train word vectors
2. Experiment: train word vectors (but ceil has no gradient ... how do we get
               around this?)
3. Experiment: train word vectors after N epochs, N a hyperparameter

## Transformation
1. Experiment: add a bias
2. Experiment: add a non-linearity

## Optimization
1. Experiment: tune batch size
2. Experiment: tune learning rate
3. Experiment: change optimizer
4. Experiment: add regularization
5. Experiment: add dropout 

## Evaluation
1. Print training error (after N epochs)

## Debugging
1. Verify that the data is correctly split into training and development
   subsets in util.py. The scores we obtain are suspiciously low, though
   the training loss is lower than it was before we implemented the hard
   separation of body identifiers between train and dev.
   


