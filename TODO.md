## Training word vectors
1. Experiment: do not train word vectors
2. Experiment: train word vectors
3. Experiment: train word vectors after N epochs, N a hyperparameter

## Transformation
1. Experiment: add a bias
2. Experiment: add a non-linearity

## Optimization
1. Experiment: tune batch size
2. Experiment: tune learning rate
3. Experiment: change optimizer

## Debugging
1. Is a bug causing our model to output prediction == UNRELATED for every
   example, or is this result symptomatic of a problem with our model 
   architecture? Possible bugs: loss is not configured correctly, feed input
   to labels_placeholders is always UNRELATED, variables are not being trained,
   tf.argmax is taken across wrong axis or applied to the wrong object or
   returns something that we do not expect


