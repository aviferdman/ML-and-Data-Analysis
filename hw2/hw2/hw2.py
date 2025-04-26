import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    # Extract labels from the last column
    labels = data[:, -1]
    
    # Calculate probability of each class
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    
    # Calculate gini impurity: 1 - Σ(p_i)²
    gini = 1 - np.sum(probabilities ** 2)
    
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    # Extract labels from the last column
    labels = data[:, -1]
    
    # Calculate probability of each class
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    
    # Calculate entropy: -Σ(p_i * log₂(p_i))
    # Handle case where p_i = 0 (log(0) is undefined)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    return entropy

class DecisionNode:

    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    def depth(self):
        """
        Calculate the depth of the tree rooted at this node.
        
        Returns:
        - depth: The maximum depth of the tree
        """
        if not self.children:  # Leaf node
            return self.depth
            
        # Return maximum depth among children
        return max(child.depth() for child in self.children)
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        # Get labels from the last column
        labels = self.data[:, -1]
        
        # Find most common label
        unique_labels, counts = np.unique(labels, return_counts=True)
        pred = unique_labels[np.argmax(counts)]
        
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        # Get the current node's impurity weighted by its proportion of samples
        n_node_samples = len(self.data)
        node_impurity = self.impurity_func(self.data)
        weighted_node_impurity = (n_node_samples / n_total_sample) * node_impurity
        
        # Calculate weighted sum of child impurities
        weighted_children_impurity = 0
        for child in self.children:
            n_child_samples = len(child.data)
            child_impurity = self.impurity_func(child.data)
            weighted_children_impurity += (n_child_samples / n_total_sample) * child_impurity
        
        # Feature importance is the decrease in weighted impurity
        self.feature_importance = weighted_node_impurity - weighted_children_impurity
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        n_samples = len(self.data)
        parent_impurity = self.impurity_func(self.data)
        
        # Split data into groups based on feature values
        groups = {}
        unique_values = np.unique(self.data[:, feature])
        for value in unique_values:
            groups[value] = self.data[self.data[:, feature] == value]
        
        # Calculate weighted sum of children impurities
        weighted_child_impurity = 0
        for value, group in groups.items():
            n_group = len(group)
            if n_group > 0:
                weighted_child_impurity += (n_group / n_samples) * self.impurity_func(group)
        
        # Calculate goodness of split
        goodness = parent_impurity - weighted_child_impurity
        
        # If using gain ratio, adjust the goodness value
        if self.gain_ratio and goodness > 0:
            # Calculate split information
            split_info = 0
            for value, group in groups.items():
                prop = len(group) / n_samples
                if prop > 0:  # Avoid log(0)
                    split_info -= prop * np.log2(prop)
            
            # Prevent division by zero
            if split_info > 0:
                goodness = goodness / split_info
            else:
                goodness = 0
                
        return goodness, groups
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        # Check max depth constraint
        if self.depth >= self.max_depth:
            self.terminal = True
            return

        n_samples, n_features = self.data.shape
        n_features = n_features - 1  # Exclude label column
        
        # Find best feature to split on
        best_feature = -1
        best_goodness = -1
        best_groups = None
        
        for feature in range(n_features):
            goodness, groups = self.goodness_of_split(feature)
            if goodness > best_goodness:
                best_goodness = goodness
                best_feature = feature
                best_groups = groups
        
        # If no improvement or pure node, make it terminal
        if best_goodness <= 0 or self.impurity_func(self.data) == 0:
            self.terminal = True
            return
            
        # Calculate chi-square value for pruning
        if self.chi < 1:  # Skip if chi=1 (no pruning)
            # Get observed frequencies
            labels = self.data[:, -1]
            unique_labels = np.unique(labels)
            n_classes = len(unique_labels)
            
            # Create contingency table
            contingency = np.zeros((len(best_groups), n_classes))
            for i, (value, group) in enumerate(best_groups.items()):
                for j, label in enumerate(unique_labels):
                    contingency[i, j] = np.sum(group[:, -1] == label)
            
            # Calculate expected frequencies
            row_totals = contingency.sum(axis=1)[:, np.newaxis]
            col_totals = contingency.sum(axis=0)
            expected = np.outer(row_totals, col_totals) / n_samples
            
            # Calculate chi-square statistic
            chi_square = np.sum(((contingency - expected) ** 2) / (expected + 1e-10))
            
            # Get degrees of freedom
            df = (len(best_groups) - 1) * (n_classes - 1)
            if df in chi_table and self.chi in chi_table[df]:
                critical_value = chi_table[df][self.chi]
                if chi_square < critical_value:
                    self.terminal = True
                    return
        
        # Create children nodes
        self.feature = best_feature
        for value, group in best_groups.items():
            child = DecisionNode(
                data=group,
                impurity_func=self.impurity_func,
                depth=self.depth + 1,
                chi=self.chi,
                max_depth=self.max_depth,
                gain_ratio=self.gain_ratio
            )
            self.add_child(child, value)
            
        # Calculate feature importance
        self.calc_feature_importance(n_samples)
        
        # Recursively split children
        for child in self.children:
            child.split()
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

                    
class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy
        
    def depth(self):
        return self.root.depth()

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return chi_training_acc, chi_testing_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






