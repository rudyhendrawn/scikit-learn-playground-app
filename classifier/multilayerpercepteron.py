import streamlit as st
from sklearn.neural_network import MLPClassifier

class NeuralNetwork():
    """
    Constructor
    ----------
    hidden_layer_sizes  : tuple, length = n_layers - 2, default (100,)
        The ith element represents the number of neurons in the ith hidden layer.
        For example, (20, 10) would mean that the MLP has two hidden layers,
        the first with 20 neurons and the second with 10.
    
    activation : `{'identity', 'logistic', 'tanh', 'relu'}`, default `'relu'`
        Activation function for the hidden layer. If you don't specify anything, no activation is applied.
        - `'identity'`, no-op activation, useful to implement linear bottleneck, returns $f(x) = x$.
        - `'logistic'`, the logistic sigmoid function, returns $f(x) = 1 / (1 + exp(-x))$.
        - `'tanh'`, the hyperbolic tan function, returns $f(x) = tanh(x)$.
        - `'relu'`, the rectified linear unit function, returns $f(x) = max(0, x)$.
    
    solver : `{'lbfgs', 'sgd', 'adam'}`, default `'adam'`
        The solver for weight optimization.
        - `'lbfgs'` is an optimizer in the family of quasi-Newton methods.
        - `'sgd'` refers to stochastic gradient descent.
        - `'adam'` refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
    
    alpha : float, optional, default 0.0001
        L2 penalty (regularization term) parameter.
    
    batch_size : int, optional, default `auto`
        Size of minibatches for stochastic optimizers. If the solver is `'lbfgs'`, the classifier will not use minibatch.
        When set to `'auto'`, `batch_size=min(200, n_samples)`.
    
    learning_rate : {'constant', 'invscaling', 'adaptive'}, default 'constant'
        Learning rate schedule for weight updates.
        - `'constant'` is a constant learning rate given by `'learning_rate_init'`.
        - `'invscaling'` gradually decreases the learning rate `'learning_rate_'` at each time step `t` using an inverse exponent
            of `power_t`. Effective learning rate is `learning_rate_init / pow(t, power_t)`.
        - `'adaptive'` keeps the learning rate constant to `'learning_rate_init'` as long as training loss keeps decreasing. 
            Each time two consecutive epochs fail to decrease training loss by at least tol, 
            or fail to increase validation score by at least tol if `early_stopping` is on, the current learning rate is divided by 5.
        Only used when `solver='sgd'`. 
    
    learning_rate_init : double, optional, default 0.001
        The initial learning rate used. It controls the step-size in updating the weights. Only used when `solver='sgd'` or `solver='adam'`.   
    
    max_iter : int, optional, default 200
        Maximum number of iterations. The solver iterates until convergence (determined by `tol`) or this number of iterations.
    
    shuffle : bool, optional, default True
        Whether to shuffle samples in each iteration. Only used when `solver='sgd'`.
    
    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; 
        If None, the random number generator is the RandomState instance used by `np.random`.
    
    verbose : int, default 0
        Controls the verbosity of the optimization algorithm.
    
    momentum : float, default 0.9
        Momentum for gradient descent update. Only used when `solver='sgd'`.
    
    nesterovs_momentum : boolean, default True
        Whether to use Nesterov's momentum. Only used when `solver='sgd'` and `momentum > 0`.
    
    early_stopping : bool, default False
        Whether to use early stopping to terminate training when validation score is not improving. 
        If set to true, it will automatically set aside 10% of training data as validation and terminate training 
        when validation score is not improving by at least tol for two consecutive epochs.
        Only effective when `solver='sgd'` or `solver='adam'`.
    
    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1.
    
    Note: Not all parameters are implemented.
    
    Ref: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    """
    def __init__(self) -> None:
        st_tuple = st.number_input('Hidden Layer Size (length = number_of_layer - 2)', value=100)
        self.hidden_layer_sizes = (st_tuple,)
        self.activation = st.selectbox('Activation Function', ['identity', 'logistic', 'tanh', 'relu'])
        self.solver = st.selectbox('Solver', ['lbfgs', 'sgd', 'adam'])
        self.alpha = st.number_input('L2 Penalty (regularization term) parameter', value=0.0001)
        self.batch_size = st.selectbox('Batch Size', ['auto', 100, 500, 1000])
        
        if self.solver == 'sgd':
            self.learning_rate = st.selectbox('Learning Rate', ['constant', 'invscaling', 'adaptive'])
        else:
            self.learning_rate = 'constant'

        if self.solver == 'sgd' or self.solver == 'adam':
            self.learning_rate_init = st.number_input('Learning Rate Initial', value=0.001)
        else:
            self.learning_rate_init = 0.001
           
        self.max_iter = st.number_input('Maximum Number of Iterations', value=200)
        self.shuffle = st.checkbox('Shuffle', value=True)
        self.random_state = 123
        self.verbose = st.checkbox('Verbose', value=False)      

        if self.solver == 'sgd':
            self.momentum = st.number_input('Momentum', value=0.9)
            self.nesterovs_momentum = st.checkbox('Nesterovs Momentum', value=True)
        else:
            self.momentum = 0.9
            self.nesterovs_momentum = True
        
        if self.solver == 'sgd' or self.solver == 'adam':
            # Only effective when `solver=sgd` or `solver=adam`
            self.early_stopping = st.checkbox('Early Stopping', value=True) 
        else:
            self.early_stopping = False
        
        if self.early_stopping == True:
            # Used if early_stopping is True.
            self.validation_fraction = st.slider('Validation Fraction', min_value=0.1, max_value=0.9, value=0.1, step=0.1)
        else:
            self.validation_fraction = 0.1

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
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes, 
            activation=self.activation, 
            solver=self.solver, 
            alpha=self.alpha,
            batch_size=self.batch_size, 
            learning_rate=self.learning_rate, 
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter, 
            shuffle=self.shuffle, 
            random_state=self.random_state, 
            verbose=self.verbose,
            momentum=self.momentum, 
            nesterovs_momentum=self.nesterovs_momentum, 
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction
        )
        self.model.fit(train_features, train_labels)
        
        return self.model.predict(test_features)