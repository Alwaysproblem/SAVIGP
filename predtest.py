import numpy as np

from experiments import data_source
from experiments import data_transformation
from savigp.kernel import ExtRBF
from savigp.likelihood import UnivariateGaussian, SoftmaxLL
from savigp import Savigp
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# define an automatic obtain kernel function.
def get_kernels(input_dim, num_latent_proc, variance = 1, lengthscale = None, ARD = False):
    return [ExtRBF(
                input_dim, variance = variance, 
                lengthscale=np.array((1.,)) if lengthscale is None else lengthscale, 
                ARD=ARD
            )
            for _ in range(num_latent_proc)]

# Load the boston dataset.
dataS = data_source.mnist_data()[0]
dim = dataS['train_outputs'].shape[1]
# dim = 1

data_pca = {}
comp_dims = 20

print("compressed with PCA")
pca = PCA(comp_dims)
pca.fit(dataS["train_inputs"])
data_pca["train_inputs"] = pca.transform(dataS["train_inputs"])
data_pca["test_inputs"] = pca.transform(dataS["test_inputs"])
data_pca["train_outputs"] = dataS["train_outputs"]
data_pca["test_outputs"] = dataS["test_outputs"]
# data_pca["train_outputs"] = np.argmax(dataS["train_outputs"], axis=1).reshape(-1,1)
# data_pca["test_outputs"] = np.argmax(dataS["test_outputs"], axis=1).reshape(-1,1)



data = data_pca

print(data["train_inputs"][0])
print(data["train_outputs"][0])

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
# kernel = [ExtRBF(data['train_inputs'].shape[1],
#                  variance=1.0,
#                  lengthscale=np.array([1.0]),
#                  ARD=False)]

kernel = get_kernels(data["train_inputs"].shape[1], dim, variance=11, lengthscale=np.array((9.,)))
# kernel = [ExtRBF(data['train_inputs'].shape[1], variance=11, lengthscale=np.array((9.,)), ARD=False)
#             for _ in range(10)]

# Set the number of inducing points to be half of the training data.
# num_inducing = int(0.5 * data['train_inputs'].shape[0])
num_inducing = int(0.007 * data['train_inputs'].shape[0])

# Transform the data before training.
print("data transformation")
# transform = data_transformation.MeanTransformation(data['train_inputs'], data['train_outputs'])
transform = data_transformation.IdentityTransformation(data['train_inputs'], data['train_outputs'])
train_inputs = transform.transform_X(data['train_inputs'])
train_outputs = transform.transform_Y(data['train_outputs'])
test_inputs = transform.transform_X(data['test_inputs'])

# Initialize the model.
print("initialize the model")
gp = Savigp(
            
            likelihood=likelihood,
            kernels=kernel,
            num_inducing=num_inducing,
            random_inducing=True,
            debug_output=True
        )

# Now fit the model to our training data.
print("fitting")
gp.fit(
        train_inputs, 
        train_outputs, 
        optimization_config={'hyp': 15}, 
        max_iterations=300, 
        optimize_stochastic=False
    )

# Make predictions. The third output is NLPD which is set to None unless test outputs are also
# provided.
print("make a prediction")
train_pred, _, _ = gp.predict(data['train_inputs'])
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
# train_acc = accuracy_score(np.argmax(train_pred, axis=1), data["train_outputs"])
train_acc = accuracy_score(np.argmax(train_pred, axis=1), np.argmax(data["train_outputs"], axis=1))
dev_acc = accuracy_score(np.argmax(test_outputs, axis=1), np.argmax(predicted_mean, axis=1))
# dev_acc = accuracy_score(test_outputs, np.argmax(predicted_mean, axis=1))

print(f"the accuracy in the training set is {train_acc * 100}%")
print(f"the accuracy in dev set is {dev_acc * 100}%")

# print(f"the accuracy in the training set is {train_acc}")
# print(f"the accuracy in dev set is {dev_acc}")
