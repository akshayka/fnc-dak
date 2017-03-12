import logging
import unittest

import util


logging.basicConfig(format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.DEBUG)


class TestLoadAndPreprocessFNCData(unittest.TestCase):
    def setUp(self):
        self.fnc_data = TestLoadAndPreprocessFNCData.fnc_data
        self.fnc_data_train = TestLoadAndPreprocessFNCData.fnc_data_train
        self.fnc_data_test = TestLoadAndPreprocessFNCData.fnc_data_test
        self.train_test_split = TestLoadAndPreprocessFNCData.train_test_split


    @classmethod
    def setUpClass(cls):
        super(TestLoadAndPreprocessFNCData, cls).setUpClass()
        tb = open("fnc-1-data/train_bodies.csv", "rb")
        ts = open("fnc-1-data/train_stances.csv", "rb")
        cls.train_test_split = 0.8
        cls.fnc_data, cls.fnc_data_train, cls.fnc_data_test = \
            util.load_and_preprocess_fnc_data(train_bodies_fstream=tb,
            test_bodies_fstream=ts, train_test_split=cls.train_test_split)
        tb.close()
        ts.close()


    def test_fnc_data_lens_are_equal(self):
        def assert_fnc_data_lens_are_equal(data, set_type):
            num_headlines = len(data.headlines)
            num_bodies = len(data.bodies)
            num_stances = len(data.stances)
            self.assertTrue(num_headlines == num_bodies == num_stances,
                "inconsistent data field sizes for %s: "
                "num_headlines %d, num_bodies %d, num_stances %d" % (
                set_type, num_headlines, num_bodies, num_stances))
        assert_fnc_data_lens_are_equal(self.fnc_data, "fnc_data")
        assert_fnc_data_lens_are_equal(self.fnc_data_train, "fnc_data_train")
        assert_fnc_data_lens_are_equal(self.fnc_data_test, "fnc_data_test")


    def test_train_test_split(self):
        training_fraction = (
            float(len(self.fnc_data_train.headlines)) /
            len(self.fnc_data.headlines)
        )

        self.assertTrue(training_fraction < self.train_test_split + 0.05 and
            training_fraction > self.train_test_split - 0.05,
            "Invalid train_test_split: expected %.2f, got %.2f; "
            "%d total examples, %d training examples" % (
            self.train_test_split, training_fraction,
            len(self.fnc_data.headlines), len(self.fnc_data_train.headlines)))

        self.assertEqual(len(self.fnc_data.headlines),
            len(self.fnc_data_train.headlines) +
            len(self.fnc_data_test.headlines),
            "Length of training data (%d) plus length of test data (%d) "
            "does not equal length of total data (%d)" % (
             len(self.fnc_data_train.headlines),
             len(self.fnc_data_test.headlines),
             len(self.fnc_data.headlines)))


    def test_distinct_bodies(self):
        train_bodies = set([''.join(b) for b in self.fnc_data_train.bodies])
        test_bodies = set([''.join(b) for b in self.fnc_data_test.bodies])
        for i, b in enumerate(train_bodies):
            self.assertTrue(b not in test_bodies, "Found body that is present "
            "in both training and testing data, %s" % b)


    def test_train_test_consistent_with_data(self):
        train_rows = set(zip(
            ["".join(h) for h in self.fnc_data_train.headlines],
            ["".join(b) for b in self.fnc_data_train.bodies],
            self.fnc_data_train.stances))
        test_rows = set(zip(
            ["".join(h) for h in self.fnc_data_test.headlines],
            ["".join(b) for b in self.fnc_data_test.bodies],
            self.fnc_data_test.stances))
        data_rows = set(zip(
            ["".join(h) for h in self.fnc_data.headlines],
            ["".join(b) for b in self.fnc_data.bodies],
            self.fnc_data.stances))
        for row in train_rows:
           self.assertTrue(row in data_rows,
            "row in training set not consistent with data, %s" % str(row))
        for row in test_rows:
           self.assertTrue(row in data_rows,
            "row in test set not consistent with data, %s" % str(row))


#    inv_body_map = {"".join(body_map[i]) : i for i in body_map}
#    original_ids = [inv_body_map["".join(b)] for b in fnc_data.bodies]
#    import pdb; pdb.set_trace()
#    assert original_ids == body_ids
