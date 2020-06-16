import numpy as np

def initialize_weights(layer_dims, method):
    np.random.seed(1)
    L = len(layer_dims)
    param = {}
    
    for l in range(1,L):
        param['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])
        param['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        #Xavier initialization
        if method == 'Xavier':
            param['W' + str(l)] = param['W' + str(l)] * np.sqrt(1 / layer_dims[l-1])
            
        #He initialization
        if method == 'He':
            param['W' + str(l)] = param['W' + str(l)] * np.sqrt(2 / layer_dims[l-1])
            
        #Bengio initialization
        if method == 'Bengio':
            param['W' + str(l)] = param['W' + str(l)] * np.sqrt(2 / (layer_dims[l] + layer_dims[l-1]))

        #Random initialization            
        if method == 'Random':
            param['W' + str(l)] = param['W' + str(l)] * 0.01
            
    return param

def linear_forward(Aprev, W, b):
    """
    Compute the linear part "Z" of the forward propagation
    """
    np.random.seed(1)
    Z = np.dot(W, Aprev) + b
    cache = (Aprev, W, b)
    assert(Z.shape == (W.shape[0], Aprev.shape[1]))
    return Z, cache

def sigmoid(Z):
    """
    Sigmoid activation function
    """
    return 1 / (1+np.exp(-Z))

def relu(Z):
    """
    ReLU activation function
    """
    return np.maximum(0, Z)

def activation_forward(Aprev, W, b, activation):
    """
    Compute activation values "A" of the forward propagation
    """
    np.random.seed(1)
    Z, linear_cache = linear_forward(Aprev, W, b)
    if activation=='sigmoid':
        A = sigmoid(Z)
    if activation=='relu':
        A = relu(Z)        
    cache = (linear_cache, Z)
    assert(A.shape == Z.shape)
    return A, cache

def forward_prop(X, param, layer_size):
    """
    Forward propagation process
    """
    np.random.seed(1)
    caches = []
    Aprev = X
    L = len(layer_size)
    for l in range(1,L-1):
        A, cache = activation_forward(Aprev, param['W' + str(l)], param['b' + str(l)], 'relu')
        Aprev = A
        caches.append(cache)
    A, cache = activation_forward(Aprev, param['W' + str(L-1)], param['b' + str(L-1)], 'sigmoid')
    caches.append(cache)
    return A, caches

def linear_backward(dZ, cache):
    """
    Compute the linear part of the backward propagation
    """
    np.random.seed(1)
    Aprev = cache[0]
    W = cache[1]
    m = Aprev.shape[1]
    dW = 1/m * np.dot(dZ, Aprev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dAprev = np.dot(W.T, dZ)
    cache = (dAprev, dW, db)
    return cache

def sigmoid_backward(Z):
    """
    Derivative of the sigmoid activation function (use in backward propagation)
    """
    A = 1 / (1+np.exp(-Z))
    return np.multiply(A, (1 - A))

def relu_backward(Z):
    """
    Derivative of the ReLU activation function (use in backward propagation)
    """
    A = np.maximum(0, Z)
    A[A>0] = 1
    return A

def activation_backward(dA, cache, activation):
    """
    Compute the backward propagation for the LINEAR->ACTIVATION layer
    """
    np.random.seed(1)
    Z = cache[1]
    if activation=='sigmoid':
        dZ = dA * sigmoid_backward(Z)
    if activation=='relu':
        dZ = dA * relu_backward(Z)
    linear_cache = linear_backward(dZ, cache[0])
    return linear_cache

def backward_prop(Y, A, cache_fwd, layer_size):
    """
    Backward propagation process
    """
    np.random.seed(1)
    grads = {}    
    L = len(layer_size)
    dA = -(np.divide(Y, A) - np.divide(1-Y, 1-A))
    l = L -1
    cache = activation_backward(dA, cache_fwd[l-1], 'sigmoid')
    dA = cache[0]
    dW = cache[1]
    db = cache[2]
    grads['dW' + str(l)] = dW
    grads['db' + str(l)] = db
    
    for l in range(L-2,0,-1):
        cache = activation_backward(dA, cache_fwd[l-1], 'relu')
        dA = cache[0]
        dW = cache[1]
        db = cache[2]
        grads['dW' + str(l)] = dW
        grads['db' + str(l)] = db
    return grads

def update_params(param, grads, layer_size, alpha=0.001):
    """
    Update weight and intercept parameters
    """
    L = len(layer_size)
    for l in range(1,L):
        param['W' + str(l)] = param['W' + str(l)] - alpha * grads['dW' + str(l)]
        param['b' + str(l)] = param['b' + str(l)] - alpha * grads['db' + str(l)]
    return param

def cost(A, Y):
    """
    Calculate the cost function across training examples
    """
    m = Y.shape[1]
    J = -1/m * (np.dot(Y, np.log(A).T) + np.dot(1-Y, np.log(1-A).T))
    J = np.squeeze(J)
    return np.round(J, 3)

def predict(A, Y):
    pred = A.copy()
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    acc = np.sum(pred==Y) / Y.shape[1]
    return pred, acc
