import unittest

from ..dataset import StratifiedFoldSplit


class TestStratifiedFoldSplit(unittest.TestCase):
    def test_split(self):
        dataset = list(zip(range(50), (i % 2 for i in range(50))))
        last_train = set()
        last_val = set()
        for fold in range(StratifiedFoldSplit.K):
            splitter = StratifiedFoldSplit(dataset, fold)
            train = splitter.train()
            val = splitter.val()

            # check that all samples have been used
            self.assertEqual(len(train), 40)
            self.assertEqual(len(val), 10)

            # no duplicates inside each
            self.assertEqual(len(set(train)), len(train))
            self.assertEqual(len(set(val)), len(val))

            # all samples used & no duplicates
            self.assertEqual(len(set(train) | set(val)), 50)

            # check stratification
            self.assertEqual(sum(1 for i in train if i[1] == 1), 20)
            self.assertEqual(sum(1 for i in val if i[1] == 1), 5)

            # different folds
            self.assertNotEqual(set(train), last_train)
            self.assertNotEqual(set(val), last_val)
            last_train = train
            last_val = val

    def test_unequal_split(self):
        # four times as many 0 as 1 samples
        dataset = list(
            zip(range(50), (1 if i % 5 == 0 else 0 for i in range(50)))
        )
        last_train = set()
        last_val = set()
        for fold in range(StratifiedFoldSplit.K):
            splitter = StratifiedFoldSplit(dataset, fold)
            train = splitter.train()
            val = splitter.val()

            # check that all samples have been used
            self.assertEqual(len(train), 40)
            self.assertEqual(len(val), 10)

            # no duplicates inside each
            self.assertEqual(len(set(train)), len(train))
            self.assertEqual(len(set(val)), len(val))

            # all samples used & no duplicates
            self.assertEqual(len(set(train) | set(val)), 50)

            # check stratification
            self.assertEqual(sum(1 for i in train if i[1] == 1), 8)
            self.assertEqual(sum(1 for i in val if i[1] == 1), 2)

            # different folds
            self.assertNotEqual(set(train), last_train)
            self.assertNotEqual(set(val), last_val)
            last_train = train
            last_val = val

    def test_three_way_split(self):
        dataset = list(zip(range(50), (i % 2 for i in range(50))))
        last_train = set()
        last_val = set()
        last_test = set()
        for fold in range(StratifiedFoldSplit.K):
            splitter = StratifiedFoldSplit(dataset, fold, test=True)
            train = splitter.train()
            val = splitter.val()
            test = splitter.test()

            # check that all samples have been used
            self.assertEqual(len(train), 30)
            self.assertEqual(len(val), 10)
            self.assertEqual(len(test), 10)

            # no duplicates inside each
            self.assertEqual(len(set(train)), len(train))
            self.assertEqual(len(set(val)), len(val))
            self.assertEqual(len(set(test)), len(test))

            # all samples used & no duplicates
            self.assertEqual(len(set(train) | set(val) | set(test)), 50)

            # check stratification
            self.assertEqual(sum(1 for i in train if i[1] == 1), 15)
            self.assertEqual(sum(1 for i in val if i[1] == 1), 5)
            self.assertEqual(sum(1 for i in test if i[1] == 1), 5)

            # different folds
            self.assertNotEqual(set(train), last_train)
            self.assertNotEqual(set(val), last_val)
            self.assertNotEqual(set(test), last_test)
            last_train = train
            last_val = val
            last_test = test
