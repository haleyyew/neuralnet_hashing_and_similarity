import sys
import argparse
from tutorial_linear_regression import linear_model, utils
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required = True,
        choices = ["2.1", "2.2","3.1","3.2","4.1","4.3"])
    io_args = parser.parse_args()
    question = io_args.question
    figspath = "/Users/haoran/Documents/neuralnet_hashing_and_similarity/tutorial_linear_regression/"


    if question == "2.1":
        # Load the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # print(data.keys())
        # print(X[0])
        # print(y[0])

        # get the number of rows(n) and columns(d)
        n,d = X.shape
        t = Xtest.shape[0]

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2) / n
        print("Training error = ", trainError)

        # Compute test error

        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2) / t
        print ("Test error = ", testError)

        # Plot model
        plt.figure()
        plt.plot(X,y,'b.', label = "Training data")
        plt.title('Training Data')
        # Choose points to evaluate the function
        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat,yhat,'g', label = "Least squares fit")
        plt.legend(loc="best")
        figname = os.path.join(figspath,"figs","leastSquares.pdf")
        print("Saving", figname)
        plt.savefig(figname)


        ''' YOUR CODE HERE'''
        # Fit the least squares model with bias
        model = linear_model.LeastSquaresBias()
        model.fit(X, y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y) ** 2) / n
        print("Training error = ", trainError)

        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest) ** 2) / t
        print("Test error = ", testError)

        # Plot model
        plt.figure()
        plt.plot(X, y, 'b.', label="Training Data")
        plt.title('Training Data with bias')
        # Choose points to evaluate the function
        Xhat = np.linspace(np.min(X), np.max(X), 1000)[:, None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat, yhat, 'g', label="Least Squares with bias prediction")
        plt.legend()
        save_figs_path = figspath+"figs/Least_Squares_with_bias.pdf"
        plt.savefig(save_figs_path)

    elif question == "2.2":

        # loads the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        n,d = X.shape
        t = Xtest.shape[0]

        for p in range(11):
            print("p=%d" % p)

            ''' YOUR CODE HERE '''
            # Fit least-squares model
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X, y)

            # Compute training error
            yhat = model.predict(X)
            trainError = np.sum((yhat - y) ** 2) / n
            print("Training error = %.0f" % trainError)

            # Compute test error

            yhat = model.predict(Xtest)
            testError = np.sum((yhat - ytest) ** 2) / t
            print("Test error     = %.0f" % testError)

            # Plot model
            plt.figure()
            plt.plot(X,y,'b.', label = "Training data")
            plt.title('Training Data. p = {}'.format(p))
            # Choose points to evaluate the function
            Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]


            '''YOUR CODE HERE'''
            #Predict on Xhat
            yhat = model.predict(Xhat)
            plt.plot(Xhat, yhat, 'g', label="Least Squares with basis {}".format(p))

            plt.legend()
            figname = os.path.join(figspath,"figs","PolyBasis%d.pdf"%p)
            print("Saving", figname)
            plt.savefig(figname)

    elif question == "3.1":
        # loads the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        (n,d) = X.shape
        t = Xtest.shape[0]

        # Split training data into a training and a validation set
        Xtrain = X[0:n//2]
        ytrain = y[0:n//2]
        Xvalid = X[n//2: n]
        yvalid = y[n//2: n]

        # Find best value of RBF kernel parameter,
        # training on the train set and validating on the validation set

        minErr = np.inf
        for s in range(-15,16):
            sigma = 2 ** s

            # Train on the training set
            model = linear_model.LeastSquaresRBF(sigma)
            model.fit(Xtrain,ytrain)

            # Compute the error on the validation set
            yhat = model.predict(Xvalid)
            validError = np.sum((yhat - yvalid)**2)/ (n//2)
            print("Error with sigma = {:e} = {}".format( sigma ,validError))

            # Keep track of the lowest validation error
            if validError < minErr:
                minErr = validError
                bestSigma = sigma

        print("Value of sigma that achieved the lowest validation error = {:e}".format(bestSigma))

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2)/n
        print("Training error = {}".format(trainError))

        # Finally, report the error on the test set
        t = Xtest.shape[0]
        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2)/t
        print("Test error = {}".format(testError))

        # Plot model
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title('Training Data')

        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat,yhat,'g',label = "Least Squares with RBF kernel and $\sigma={}$".format(bestSigma))
        plt.ylim([-300,400])
        plt.legend()
        figname = os.path.join(figspath,"figs","least_squares_rbf.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "3.2":
        # loads the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        (n,d) = X.shape
        t = Xtest.shape[0]

        # Split training data into a training and a validation set
        Xtrain = X[0:n//2]
        ytrain = y[0:n//2]
        Xvalid = X[n//2: n]
        yvalid = y[n//2: n]

        # Find best value of RBF kernel parameter,
        # training on the train set and validating on the validation set

        minErr = np.inf
        for s in range(-15,16):
            sigma = 2 ** s

            all_data = np.hstack((X, y))
            np.random.shuffle(all_data)
            # Perform Cross Validation
            fold_length = all_data.shape[0]//10
            sum_errors = 0
            for i in range(10):
                valid_data = all_data[i*fold_length:(i+1)*fold_length][:]
                train_data = np.vstack((all_data[0:i*fold_length], all_data[(i+1)*fold_length:all_data.shape[0]]))

                Xtrain = train_data[:,0]
                Xtrain= np.reshape(Xtrain, (Xtrain.shape[0],1))
                ytrain = train_data[:,1]

                Xvalid = valid_data[:,0]
                Xvalid = np.reshape(Xvalid, (Xvalid.shape[0],1))
                yvalid = valid_data[:,1]


                # Train on the training set
                model = linear_model.LeastSquaresRBF(sigma)
                model.fit(Xtrain,ytrain)

                # Compute the error on the validation set
                yhat = model.predict(Xvalid)
                validError = np.sum((yhat - yvalid)**2)/ (n//2)
                sum_errors += validError
                print("Error with sigma = {:e} = {}".format( sigma ,validError))

            validError = sum_errors / 10
            # Keep track of the lowest validation error
            if validError < minErr:
                minErr = validError
                bestSigma = sigma

        print("Value of sigma that achieved the lowest validation error = {:e}".format(bestSigma))

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2)/n
        print("Training error = {}".format(trainError))

        # Finally, report the error on the test set
        t = Xtest.shape[0]
        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2)/t
        print("Test error = {}".format(testError))

        # Plot model
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title('Training Data')

        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat,yhat,'g',label = "Least Squares with RBF kernel and $\sigma={}$".format(bestSigma))
        plt.ylim([-300,400])
        plt.legend()
        figname = os.path.join(figspath,"figs","least_squares_rbf_cross_val.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "4.1":
        # loads the data in the form of dictionary
        data = utils.load_dataset("outliersData")
        X = data['X']
        y = data['y']

        # Plot data
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title("Training data")

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)

        # Draw model prediction
        Xsample = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xsample)
        plt.plot(Xsample,yhat,'g-', label = "Least squares fit", linewidth=4)
        plt.legend()
        figname = os.path.join(figspath,"figs","least_squares_outliers.pdf")
        print("Saving", figname)
        plt.savefig(figname)

        ''' YOUR CODE HERE '''
        # Fit weighted least-squares estimator
        z = np.concatenate(([1]*400,[0.1]*100),axis = 0)
        model = linear_model.WeightedLeastSquares()
        model.fit(X,y,z)

        # Draw model prediction
        Xsample = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xsample)
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title("Training data")
        plt.plot(Xsample,yhat,'g-', label = "Least squares fit", linewidth=4)
        plt.legend()
        figname = os.path.join(figspath,"figs","least_squares_outliers_weighted.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "4.3":
        # loads the data in the form of dictionary
        data = utils.load_dataset("outliersData")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        # Plot data
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title("Training data")

        # Draw model prediction
        Xsample = np.linspace(np.min(X), np.max(X), 1000)[:,None]
        yhat = model.predict(Xsample)
        plt.plot(Xsample, yhat, 'g-', label = "Least squares fit", linewidth=4)
        plt.legend()
        figname = os.path.join(figspath,"figs","gradient_descent_model.pdf")
        print("Saving", figname)
        plt.savefig(figname)
