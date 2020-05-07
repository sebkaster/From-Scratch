# Importing Modules
from sklearn import datasets
import matplotlib.pyplot as plt
import math
import numpy as np
from typing import Optional, List, Tuple, Union
import random


class BinaryTreeNode:
    def __init__(self, subset: np.ndarray, height: int = 1) -> None:
        self.left: Optional[BinaryTreeNode] = None
        self.right: Optional[BinaryTreeNode] = None
        self.height = height
        self.split: Optional[Tuple[int, Union[int, float]]] = None
        self.data: np.ndarray = subset


class IsolationForest:
    def __init__(self, dataset: np.ndarray, num_trees: int = 100, sub_sampling_size: int = 256) -> None:
        self.dataset = dataset
        self.num_trees = num_trees
        self.sub_sampling_size = sub_sampling_size
        self.isolation_trees: List[IsolationTree] = []
        self.create_forest()

    def create_forest(self) -> None:
        for i in range(self.num_trees):
            subset_idxs = np.random.randint(self.dataset.shape[0], size=self.sub_sampling_size)
            subset = self.dataset[subset_idxs, :]
            i_Tree = IsolationTree(subset)
            self.isolation_trees.append(i_Tree)

    def check_anomaly(self, data_point: np.ndarray) -> bool:
        lambda_func = lambda tree: tree.traverse_tree(data_point)
        mean_height = np.mean(list(map(lambda_func, self.isolation_trees)))

        # Euler-Mascheroni constant
        gamma = 0.57721566490153286060651209008240243104215933593992
        if self.sub_sampling_size > 2:
            c_m = 2.0 * math.log(self.sub_sampling_size - 1) + gamma - (
                    (2.0 * self.sub_sampling_size) / self.dataset.shape[0])
        elif self.sub_sampling_size == 2:
            c_m = 1
        else:
            c_m = 0

        return 2.0 ** (-mean_height / c_m)


class IsolationTree:
    def __init__(self, subset: np.ndarray) -> None:
        self.max_height: int = math.ceil(math.log2(subset.shape[0]))
        self.root = BinaryTreeNode(subset)
        self.create_tree()

    def create_tree(self) -> None:
        stack = list()
        stack.append(self.root)
        while len(stack) > 0:
            current_node = stack.pop()
            if current_node.data.shape[0] > 1 and current_node.height < self.max_height:
                random_feature = random.randint(0, current_node.data.shape[1] - 1)
                random_split = random.uniform(np.min(current_node.data[:, random_feature]),
                                              np.max(current_node.data[:, random_feature]))
                current_node.split = (random_feature, random_split)

                left_data = current_node.data[current_node.data[:, random_feature] <= random_split]
                left_node = BinaryTreeNode(left_data, current_node.height + 1)
                current_node.left = left_node
                stack.append(left_node)

                right_data = current_node.data[current_node.data[:, random_feature] > random_split]
                right_node = BinaryTreeNode(right_data, current_node.height + 1)
                current_node.right = right_node
                stack.append(right_node)

    def traverse_tree(self, data_point: np.ndarray) -> int:
        current_node = self.root
        while current_node.left and current_node.right:
            if data_point[current_node.split[0]] <= current_node.split[1]:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return current_node.height


def pprint_tree(node: BinaryTreeNode, _prefix: str = "", _last: bool = True) -> None:
    print(_prefix, "`- " if _last else "|- ", node.split, sep="")
    _prefix += "   " if _last else "|  "
    has_childs = False if node.left is None else True
    if has_childs:
        pprint_tree(node.left, _prefix, False)
        pprint_tree(node.right, _prefix, True)


rng = np.random.RandomState(42)

# Generating training data
X_train = 0.2 * rng.randn(1000, 2)
X_train = np.r_[X_train + 3, X_train]

# Generating new, 'normal' observation
X_test = 0.2 * rng.randn(200, 2)
X_test = np.r_[X_test + 3, X_test]

# Generating outliers
X_outliers = rng.uniform(low=-1, high=5, size=(50, 2))
forest = IsolationForest(X_train, 100, 128)

print(forest.check_anomaly(X_test[0]))
print(forest.check_anomaly(X_test[1]))
print(forest.check_anomaly(X_test[2]))
print(forest.check_anomaly(X_test[3]))

for i in range(X_test.shape[0]):
    print(forest.check_anomaly(X_test[i]))

print('dd')
for i in range(X_outliers.shape[0]):
    print(forest.check_anomaly(X_outliers[i]))

# print(pprint_tree(forest.isolation_trees[0].root))
