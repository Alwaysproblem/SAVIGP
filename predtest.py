import numpy as np

from experiments import data_source
from experiments import data_transformation
from savigp.kernel import ExtRBF
from savigp.likelihood import UnivariateGaussian, SoftmaxLL
from savigp import Savigp
from sklearn.metrics import accuracy_score


# Load the boston dataset.
data = data_source.mnist_data()[0]
dim = data['train_inputs'].shape[1]
# print(dim)
# data = data_source.boston_data()[0]
# print(data['test_outputs'])
# print(np.argmax(data['test_outputs'], axis = 1))


# Define a univariate Gaussian likelihood function with a variance of 1.
print("Define the softmax likelihood.")
# likelihood = UnivariateGaussian(np.array([1.0]))
likelihood = SoftmaxLL(dim)

# Define a radial basis kernel with a variance of 1, lengthscale of 1 and ARD disabled.
print("define a kernel")
kernel = [ExtRBF(data['train_inputs'].shape[1],
                 variance=1.0,
                 lengthscale=np.array([1.0]),
                 ARD=False)]

# Set the number of inducing points to be half of the training data.
num_inducing = int(0.5 * data['train_inputs'].shape[0])

# Transform the data before training.
print("data transformation")
transform = data_transformation.MeanTransformation(data['train_inputs'], data['train_outputs'])
train_inputs = transform.transform_X(data['train_inputs'])
train_outputs = transform.transform_Y(data['train_outputs'])
test_inputs = transform.transform_X(data['test_inputs'])

# Initialize the model.
print("initialize the model")
gp = Savigp(likelihood=likelihood,
            kernels=kernel,
            num_inducing=num_inducing,
            debug_output=True)

# Now fit the model to our training data.
print("fitting")
gp.fit(train_inputs, train_outputs)

# Make predictions. The third output is NLPD which is set to None unless test outputs are also
# provided.
print("make a prediction")
train_pred, _, _ = gp.predict(data['train_input'])
# predicted_mean, predicted_var, _ = gp.predict(data['test_inputs'])
predicted_mean, _, _ = gp.predict(data['test_inputs'])

# Untransform the results.
predicted_mean = transform.untransform_Y(predicted_mean)
train_pred = transform.untransform_Y(train_pred)
# predicted_var = transform.untransform_Y_var(predicted_var)

# Print the mean standardized squared error.
test_outputs = data['test_outputs']
# print("MSSE:", (((predicted_mean - test_outputs) ** 2).mean() /
#                 ((test_outputs.mean() - test_outputs) ** 2).mean()))

# Print the accuracy
# print(pre)
dev_acc = accuracy_score(np.argmax(test_outputs, axis=1), np.argmax(predicted_mean, axis=1))
# dev_acc = 
print(f"the accuracy in dev set is {dev_acc * 100}%")
