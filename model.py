import logging

from util import ConfusionMatrix, Progbar, minibatches, LBLS, RELATED, UNRELATED

import matplotlib.pyplot as plt
import tensorflow as tf

logger = logging.getLogger("baseline")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Model(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """
    def __init__(self, config, verbose=False):
        self.verbose = verbose


    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.

        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are
        used as inputs by the rest of the model building and will be fed data
        during training.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for one step of training.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If labels_batch is None, then no labels are added to feed_dict.

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_prediction_op(self):
        """Transforms a batch of input data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model. See

        Args:
            loss: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """

        raise NotImplementedError("Each Model must re-implement this method.")


    def train_on_batch(self, sess, headlines_batch, bodies_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(headlines_batch, bodies_batch, self.epoch,
            dropout=self.config.dropout, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss


    def predict_on_batch(self, sess, headlines_batch, bodies_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        # Do not apply dropout during evaluation!
        feed = self.create_feed_dict(headlines_batch, bodies_batch,
            self.epoch, dropout=0)
        predictions = sess.run(self.final_pred, feed_dict=feed)
        return predictions


    def output(self, sess, inputs):
        """
        Reports the output of the model on examples.
        """

        preds = []
        headlines, bodies, stances = zip(*inputs)
        data = zip(headlines, bodies)
        prog = Progbar(target=1 + int(len(stances) / self.config.batch_size))
        # TODO(akshayka): Verify that data is in the correct structure
        for i, batch in enumerate(minibatches(data, self.config.batch_size,
            shuffle=False)):
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
            prog.update(i + 1, [])
        return (headlines, bodies), stances, preds


    def evaluate(self, sess, examples):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and
        constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
        Returns:
            The F1 score for predicting the relationship between
            headline-body pairs
        """
        # TODO(akshayka): Implement a report that tells us the inputs
        # on which we guessed incorrectly
        token_cm = ConfusionMatrix(labels=LBLS, default_label=UNRELATED)

        correct_guessed_related, total_gold_related, total_guessed_related = (
            0., 0., 0.)
        _, labels, labels_hat = self.output(sess, examples)
        score = 0
        num_unrelated = len([l for l in labels if l == UNRELATED])
        num_related = len(labels) - num_unrelated
        unrelated_score = 0.25 * num_unrelated
        max_score = unrelated_score + 1.0 * num_related
        for l, l_hat in zip(labels, labels_hat):
            token_cm.update(l, l_hat)
            if l == l_hat:
                score += 0.25
                if l != UNRELATED:
                    score += 0.5
            if l in RELATED and l_hat in RELATED:
                score += 0.25

            if l == l_hat and l in RELATED:
                correct_guessed_related += 1
            if l in RELATED:
                total_gold_related += 1
            if l_hat in RELATED:
                total_guessed_related += 1


        p = correct_guessed_related / total_guessed_related if \
            total_guessed_related > 0 else 0
        r = correct_guessed_related / total_gold_related if \
            total_gold_related > 0 else 0

        if total_guessed_related == 0:
            logging.warn("total_guessed_related == 0!")
        if total_gold_related == 0:
            logging.warn("total_gold_related == 0!")
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        unrelated_ratio = unrelated_score / max_score
        score_ratio = score / max_score
        return token_cm, (p, r, f1), (unrelated_ratio, score_ratio)


    def run_epoch(self, sess, train_examples, dev_examples):
        def eval_helper(sess, examples):
            token_cm, entity_scores, ratios = self.evaluate(sess,
                examples)
            logger.debug("Token-level confusion matrix:\n" +
                token_cm.as_table())
            logger.debug("Token-level scores:\n" + token_cm.summary())
            logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)
            logger.info("FNC Score: %.2f", ratios[1])
            logger.info("Unrelated Score: %.2f", ratios[0])
            fnc_score = ratios[1]
            return fnc_score

        prog = Progbar(target=1 + int(len(train_examples) /
            self.config.batch_size))
        for i, batch in enumerate(minibatches(train_examples,
            self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            prog.update(i + 1, [("train loss", loss)])
        print("")

        train_fnc_score = None
        if self.verbose:
            logger.info("Evaluating on training data")
            train_fnc_score = eval_helper(sess, train_examples)

        logger.info("Evaluating on development data")
        fnc_score = eval_helper(sess, dev_examples)
        return train_fnc_score, fnc_score
        


    def fit(self, sess, saver, train_examples, dev_examples):
        best_score = 0.

        tr_scores = []
        dev_scores = []
        epochs = range(1, self.config.n_epochs + 1)
        for epoch in epochs:
            self.epoch = epoch
            logger.info("Epoch %d out of %d", self.epoch, self.config.n_epochs)
            tr_score, dev_score = self.run_epoch(sess, train_examples,
                dev_examples)
            tr_scores.append(tr_score)
            dev_scores.append(dev_score)
            if dev_score > best_score:
                best_score = dev_score
                if saver:
                    logger.info("New best score! Saving model in %s",
                        self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
        plt.figure()
        plt.plot(epochs, tr_scores, "ro", label="training scores")
        plt.plot(epochs, dev_scores, "bo", label="dev scores")
        plt.title("FNC Scores across Epochs")
        plt.legend()
        plt.savefig(self.config.output_path + "fnc_scores.png")
        return best_score
    

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        if self.config.similarity_metric:
            # TODO(akshayka): Should sim preds be integers?
            self.final_pred = tf.round(self.pred)
            self.final_pred = tf.maximum(tf.minimum(self.final_pred, 3), 0)
        else:
            self.final_pred = tf.argmax(self.pred, axis=1)
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
