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

## Debugging
1. Verify that the data is correctly split into training and development
   subsets in util.py. The scores we obtain are suspiciously low, though
   the training loss is lower than it was before we implemented the hard
   separation of body identifiers between train and dev.
   


