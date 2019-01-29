import numpy as np
import pandas as pd
import torch as tc
from torch.autograd import Variable
    
class PMF(object):
    def __init__(self, num_feat=10, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=20, num_batches=10, batch_size=1000):
        self.num_feat = num_feat  # Number of latent features,
        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)

        self.w_Item = None  # Item feature vectors
        self.w_User = None  # User feature vectors

        self.rmse_train = []
        self.rmse_test = []

    # ***Fit the model with train_tuple and evaluate RMSE on both train and test data.  ***********
    # ***************** train_vec=TrainData, test_vec=TestData*************
    def train(self, train_vec, test_vec):
        # mean subtraction
        self.mean_inv = np.mean(train_vec[:, 2])

        pairs_train = train_vec.shape[0]  # traindata
        pairs_test = test_vec.shape[0]  # testdata

        # 1-p-i, 2-m-c
        num_user = int(max(np.amax(train_vec[:, 0]), np.amax(test_vec[:, 0]))) + 1  # user
        num_item = int(max(np.amax(train_vec[:, 1]), np.amax(test_vec[:, 1]))) + 1  # movie

        incremental = False  #
        if ((not incremental) or (self.w_Item is None)):
            # initialize
            self.epoch = 0
            self.w_Item = 0.1 * np.random.randn(num_item, self.num_feat)  # M x D
            self.w_User = 0.1 * np.random.randn(num_user, self.num_feat)  # N x D

            self.w_Item_inc = np.zeros((num_item, self.num_feat))  # M x D
            self.w_User_inc = np.zeros((num_user, self.num_feat))  # N x D

        while self.epoch < self.maxepoch:
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])
            np.random.shuffle(shuffled_order)

            # Batch update
            for batch in range(self.num_batches):
                # print "epoch %d batch %d" % (self.epoch, batch+1)

                test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(test, shuffled_order.shape[0])

                batch_UserID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_ItemID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

                # Compute Objective Function
                pred_out = np.sum(np.multiply(self.w_User[batch_UserID, :],
                                              self.w_Item[batch_ItemID, :]),
                                  axis=1)  # mean_inv subtracted

                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv

                # Compute gradients
                Ix_User = 2 * np.multiply(rawErr[:, np.newaxis], self.w_Item[batch_ItemID, :]) \
                       + self._lambda * self.w_User[batch_UserID, :]
                Ix_Item = 2 * np.multiply(rawErr[:, np.newaxis], self.w_User[batch_UserID, :]) \
                       + self._lambda * (self.w_Item[batch_ItemID, :])  # np.newaxis :increase the dimension

                dw_Item = np.zeros((num_item, self.num_feat))
                dw_User = np.zeros((num_user, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_Item[batch_ItemID[i], :] += Ix_Item[i, :]
                    dw_User[batch_UserID[i], :] += Ix_User[i, :]

                # Update with momentum
                self.w_Item_inc = self.momentum * self.w_Item_inc + self.epsilon * dw_Item / self.batch_size
                self.w_User_inc = self.momentum * self.w_User_inc + self.epsilon * dw_User / self.batch_size

                self.w_Item = self.w_Item - self.w_Item_inc
                self.w_User = self.w_User - self.w_User_inc

                if batch == self.num_batches - 1:
                    # Compute Objective Function after
                    pred_out = np.sum(np.multiply(self.w_User[np.array(train_vec[:, 0], dtype='int32'), :],
                                                  self.w_Item[np.array(train_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - train_vec[:, 2] + self.mean_inv
                    #obj = np.linalg.norm(rawErr) ** 2 \
                    #      + 0.5 * self._lambda * (np.linalg.norm(self.w_User) ** 2 + np.linalg.norm(self.w_Item) ** 2)
                    obj = np.sum(rawErr**2)

                    self.rmse_train.append(np.sqrt(obj / pairs_train))

                    # Compute validation error
                    pred_out = np.sum(np.multiply(self.w_User[np.array(test_vec[:, 0], dtype='int32'), :],
                                                  self.w_Item[np.array(test_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - test_vec[:, 2] + self.mean_inv
                    #self.rmse_test.append(np.linalg.norm(rawErr) / np.sqrt(pairs_test))
                    obj = np.sum(rawErr**2)
                    self.rmse_test.append(np.sqrt(obj / pairs_test))

                    # Print info
                    print('Training RMSE: %f, Test RMSE %f' % (self.rmse_train[-1], self.rmse_test[-1]))
        
        self.prediction = self.w_User.dot(self.w_Item.T) + self.mean_inv

    # ****************Set parameters by providing a parameter dictionary.  ***********
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)
    
    
    
class LMF():
    
    def __init__(self, X, mask, rank, eta, nu, momentum, MAX_ITER):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        
        Arguments
        - X (ndarray)    : user-item rating matrix
        - Mask (ndarray) : mask matrix
        - rank (int)     : number of latent dimensions
        - eta (float)    : learning rate
        - nu (float)     : regularization parameter
        """
        
        self.X = X
        self.M, self.N = X.shape
        self.K = rank
        self.mask = mask
        self.eta = eta
        self.nu = nu
        self.momentum = momentum
        self.MAX_ITER = MAX_ITER
        
        self.W = None
        self.H = None
        
        self.error_list = []
        self.error = None

    def train(self):
        tc.manual_seed(1)
        self.W = Variable(tc.randn(self.M, self.K), requires_grad=True)
        self.H = Variable(tc.randn(self.K, self.N), requires_grad=True)
        
        opt = tc.optim.SGD([self.W, self.H], lr=self.eta, momentum=self.momentum, weight_decay=self.nu)
        
        for epoch in range(self.MAX_ITER):
            opt.zero_grad()
            E = -self.likelihood()
            E.backward()
            opt.step()
            self.error_list.append(float(E.detach().numpy()))
            
        self.error = float(E.detach().numpy())
        self.prediction = self.sigmoid(tc.matmul(self.W, self.H))

    def sigmoid(self, t):
        return 1./(1+tc.exp(-t))

    def likelihood(self):
        return tc.sum(self.X * tc.matmul(self.W, self.H) - self.mask * tc.log(1 + tc.exp(tc.matmul(self.W, self.H))))
    
    
    
class MF():
    
    def __init__(self, X, mask, rank, eta, nu, momentum, MAX_ITER):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        
        Arguments
        - X (ndarray)    : user-item rating matrix
        - Mask (ndarray) : mask matrix
        - rank (int)     : number of latent dimensions
        - eta (float)    : learning rate
        - nu (float)     : regularization parameter
        """
        
        self.X = X
        self.M, self.N = X.shape
        self.K = rank
        self.mask = mask
        self.eta = eta
        self.nu = nu
        self.momentum = momentum
        self.MAX_ITER = MAX_ITER
        
        self.W = None
        self.H = None
        
        self.error_list = []
        self.error = None
        

    def train(self):
        tc.manual_seed(1)
        self.W = Variable(tc.randn(self.M, self.K), requires_grad=True)
        self.H = Variable(tc.randn(self.K, self.N), requires_grad=True)
        
        opt = tc.optim.SGD([self.W, self.H], lr=self.eta, momentum=self.momentum,weight_decay=self.nu)
        #opt = tc.optim.Adam([self.W, self.H], lr=self.eta, weight_decay=self.nu, amsgrad=True)
        
        for epoch in range(self.MAX_ITER):
            opt.zero_grad()
            E = -self.likelihood()
            E.backward()
            opt.step()
            self.error_list.append(float(E.detach().numpy()))
            
        self.error = float(E.detach().numpy())
        self.prediction = tc.matmul(self.W, self.H)


    def likelihood(self):
        return -tc.sum((self.X - self.mask * tc.matmul(self.W, self.H))**2)/2
    
    
class NMF ():
    
    def __init__(self, X, mask, rank, MAX_ITER):
        self.X = X + 1.e-6
        self.M, self.N = X.shape
        self.K = rank
        self.mask = mask
        self.MAX_ITER = MAX_ITER
        
        self.W = 5 * np.random.rand(self.M, self.K)
        self.H = 5 * np.random.rand(self.K, self.N)
        
        self.error_list = []
        self.error = None
    
    def train(self):
        epsilon = 1.e-3
        for epoch in range(self.MAX_ITER):
            Xhat = self.W.dot(self.H)
            self.W = self.W * ((self.X / Xhat).dot(self.H.T) / (np.dot(self.mask, self.H.T) + epsilon))
            
            Xhat = self.W.dot(self.H)
            self.H = self.H * (self.W.T.dot(self.X / Xhat) / (np.dot(self.W.T, self.mask) + epsilon))
        
        self.prediction = self.W.dot(self.H)
    
class ALS ():
    def __init__(self, X, mask, rank, lambda_, MAX_ITER):
        self.X = X
        self.M, self.N = X.shape
        self.K = rank
        self.mask = mask
        self.lambda_ = lambda_
        self.MAX_ITER = MAX_ITER
        
        self.W = 5 * np.random.rand(self.M, self.K)
        self.H = 5 * np.random.rand(self.K, self.N)
        self.prediction = None
        
        self.error_list = []
    
    
    def train(self):
        for epoch in range(self.MAX_ITER):
            for u, mask_u in enumerate(self.mask):
                self.W[u] = np.linalg.solve(np.dot(self.H, np.dot(np.diag(mask_u), self.H.T)) + self.lambda_ * np.eye(self.K),
                                            np.dot(self.H, np.dot(np.diag(mask_u), self.X[u].T))).T
            for i, mask_i in enumerate(self.mask.T):
                self.H[:,i] = np.linalg.solve(np.dot(self.W.T, np.dot(np.diag(mask_i), self.W)) + self.lambda_ * np.eye(self.K),
                                              np.dot(self.W.T, np.dot(np.diag(mask_i), self.X[:, i])))
            
            self.error_list.append(self.get_error())
            print('{}th iteration is completed'.format(epoch))     
        self.prediction = np.dot(self.W, self.H)
        
    def get_error(self):
        return np.sum((self.mask * (self.X - np.dot(self.W, self.H)))**2)

def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.User_ID == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'Movie_ID', right_on = 'Movie_ID').
                     sort_values(['Rating'], ascending=False)
                 )

    print ('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print ('Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations))
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['Movie_ID'].isin(user_full['Movie_ID'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'Movie_ID',
               right_on = 'Movie_ID').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations


class MF_foreigner():
    
    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        
        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """
        
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        
        self.training_process = []

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
        
        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        
        # Perform stochastic gradient descent for number of iterations
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            self.training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))
        

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            
            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction
    
    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)






def topK(prediction, test_vec, k=100):
    inv_lst = np.unique(test_vec[:, 0].astype(int))
    pred = {}
    for inv in inv_lst:
        if pred.get(inv, None) is None:
            pred[inv] = np.argsort(prediction[inv,:])[-k:]

    intersection_cnt = {}
    for i in range(test_vec.shape[0]):
        if test_vec[i, 1] in pred[test_vec[i, 0]]:
            intersection_cnt[test_vec[i, 0]] = intersection_cnt.get(test_vec[i, 0], 0) + 1
    invPairs_cnt = np.bincount(np.array(test_vec[:, 0], dtype='int32'))

    precision_acc = 0.0
    recall_acc = 0.0
    for inv in inv_lst:
        precision_acc += intersection_cnt.get(inv, 0) / float(k)
        recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])

    return precision_acc / len(inv_lst), recall_acc / len(inv_lst)