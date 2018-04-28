
import numpy as np
from math import cos, sin, exp, pi


def myGarborKernel(ksize, sigma, theta, Lambda, psi, gamma, ktype):
    k = (ksize - 1) / 2
    odd_kernel = np.zeros((ksize,ksize), np.float32)
    even_kernel = np.zeros((ksize, ksize), np.float32)
    for i in range(-k, k+1):
        for j in range(-k, k+1):
            a = i * cos(theta) + j * sin(theta)
            b = j * cos(theta) - i * sin(theta)
            odd_resp = exp(-1.0/8.0/sigma/sigma*(4.0*a*a+b*b)) * sin(2.0*pi*a/Lambda)
            even_resp = exp(-1.0/8.0/sigma/sigma*(4.0*a*a+b*b)) * cos(2.0*pi*a/Lambda)
            odd_kernel[i+k, j+k] = odd_resp
            even_kernel[i+k, j+k] = even_resp

    odd_mean = odd_kernel.mean()
    even_mean = even_kernel.mean()
    odd_kernel = odd_kernel - odd_mean
    even_kernel = even_kernel - even_mean

    odd_square_sum = np.sum(odd_kernel**2)
    even_square_sum = np.sum(even_kernel**2)
    l2_sum_odd = odd_square_sum / (17.0*17.0)
    l2_sum_even = even_square_sum / (17.0*17.0)

    odd_kernel = odd_kernel / l2_sum_odd
    even_kernel = even_kernel / l2_sum_even

    return [odd_kernel, even_kernel]








