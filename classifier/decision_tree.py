# Create class that implement sklearn DecisionTreeClassifier
# The class could customize the parameters of the DecisionTreeClassifier
# then implement the customization to streamlit UI
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

class DecisionTree:
    """
    Constructors
    ------------
    criterion : string, optional (default=”gini”)
        The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
    
    splitter : string, optional (default=”best”)
        The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.
    
    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    
    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.
    
    min_samples_leaf : int, optional (default=1)
        The minimum number of samples required to be at a leaf node.
    
    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the input samples required to be at a leaf node.
    
    max_features : int, float, string or None, optional (default=”auto”)
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
            `int(max_features * n_features)` features are considered at each split.
        - If `'auto'`, then `max_features=sqrt(n_features)`.
        - If `'sqrt'`, then `max_features=sqrt(n_features)`.
        - If `'log2'`, then `max_features=log2(n_features)`.
        - If `'None'`, then `max_features=n_features`.
    
    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with max_leaf_nodes in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    
    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    
    class_weight : dict, list of dicts, “balanced” or None, optional (default=None)
        Weights associated with classes in the form {class_label: weight}.
        If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
        Note that for multioutput (including multilabel) weights should be defined for each class of every column in its own dict. For example, for four-class multilabel classification weights should be [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of [{1:1}, {2:5}, {3:1}, {4:1}].
        The `'balanced'` mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as `n_samples / (n_classes * np.bincount(y))`.
        For multi-output, the weights of each column of y will be multiplied.
    
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. 
        By default, no pruning is performed. 

    Note: Not all parameters are implemented.
    Ref: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#
    """
    def __init__(self) -> None:
        col1, col2 = st.columns(2)
        
        with col1:
            self.criterion = st.selectbox('Criterion', ['gini', 'entropy'], index=0)
            self.splitter = st.selectbox('Splitter', ['best', 'random'], index=0)
            self.max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=10, step=1)
            self.min_samples_split = st.slider('Min Samples Split', min_value=2, max_value=20, value=2, step=1)
            self.min_samples_leaf = st.number_input('Min Samples Leaf', value=2)
            self.min_weight_fraction_leaf = st.slider('Min Weight Fraction Leaf', min_value=0.0, max_value=0.9, value=0.0, step=0.1)
        with col2:
            self.max_features = st.selectbox('Max Features', ['auto', 'sqrt', 'log2', 'None'], index=0)
            self.random_state = st.number_input('Random State', value=123)
            self.max_leaf_nodes = st.slider('Max Leaf Nodes', min_value=2, max_value=20, value=2, step=1)
            self.min_impurity_decrease = st.slider('Min Impurity Decrease', min_value=0.0, max_value=1.0, value=0.0, step=0.1)
            self.ccp_alpha = st.slider('CCP Alpha', min_value=0.0, max_value=1.0, value=0.0, step=0.1)
            # self.class_weight = st.sidebar.slider('Class Weight', 0, 0.5, 0)
        self.model = None

    def fit_and_predict(self, train_features, train_labels, test_features) -> object:
        """
        Parameters
        ----------
        train_features : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        
        train_labels : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in regression).
        
        test_features : array-like or sparse matrix of shape = [n_samples, n_features]
            The test input samples.
        """
        self.model = DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha
        )
        self.model.fit(train_features, train_labels)
        
        return self.model.predict(test_features)
        
