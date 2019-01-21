# import numpy as np
# from savigp.kernel import ExtRBF
# from savigp.likelihood import UnivariateGaussian
# from savigp.full_gaussian_process import FullGaussianProcess
# from savigp.diagonal_gaussian_process import DiagonalGaussianProcess
# from savigp.optimizer import batch_optimize_model
# from savigp import model_logging


re = []

# def fun(arg):
#     a, b = arg
#     print(a, b)
#     re.append(a+b)
#     # return a+b

# a = list(map(fun, [(1,2), (3,4), (5,6)]))

a = map(sum, [(1,2)])

print a
# print(re)