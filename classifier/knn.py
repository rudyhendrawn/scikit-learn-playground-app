import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

class KNN:
    """
    Constructors
    ----------
    
    n_neighbors : int, default = 5
        Number of neighbors to use by default for :meth:`k_neighbors` queries.
    
    weights : str or callable
        Weight function used in prediction. Possible values:
        - `'uniform'`: uniform weights.  All points in each neighborhood are weighted equally.
        - `'distance'`: weight points by the inverse of their distance. In this case, 
            closer neighbors of a query point will have a greater influence than neighbors which are further away.
    
    algorithm: {`auto`, `ball_tree`, `kd_tree`, `brute`}, default = `auto`
        Algorithm used to compute the nearest neighbors:
        - `'ball_tree'` will use :class:`BallTree`
        - `'kd_tree'` will use :class:`KDTree`
        - `'brute'` will use a brute-force search.
        - `'auto'` will attempt to decide the most appropriate algorithm based on the values passed to :meth:`fit` method.
    
    leaf_size : int, default = 30
        Leaf size passed to :class:`BallTree` or :class:`KDTree`. This can affect the speed of the construction and query, as well as the memory required to store the tree. 
        The optimal value depends on the nature of the problem.
    
    power_parameter : float, default = 2
        Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), 
        and euclidean_distance (l2) for p = 2. For arbitrary p, Minkowski distance (l_p) is used.
    
    Note: Not all parameters are implemented.        
    
    Ref: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
    """
    def __init__(self) -> None:
        self.n_neighbors = st.slider('Number of neighbors', min_value=5, max_value=50, value=5, step=1)
        self.weights = st.selectbox('Weight', ('uniform', 'distance'), index=0)
        self.algorithm = st.selectbox('Algorithm', ('auto', 'ball_tree', 'kd_tree', 'brute'), index=0)
        self.leaf_size = st.slider('Leaf Size', min_value=1, max_value=50, value=30, step=1)
        self.power_parameter = st.slider('Power Parameter', min_value=1, max_value=5, value=2, step=1)
        self.metric = st.selectbox('Metric', ('minkowski', 'euclidean', 'manhattan'), index=0)
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
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.power_parameter,
            metric=self.metric
        )        
        self.model.fit(train_features, train_labels)
        
        return self.model.predict(test_features)


    