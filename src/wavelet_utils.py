""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import numpy as np

def unroll_shape(c):
    if isinstance(c, tuple):
        return tuple([unroll_shape(cc) for cc in c])
    elif isinstance(c, list):
        return [unroll_shape(cc) for cc in c]
    else:
        return c.shape

def coeffs_to_array(coeffs):

    shapes = unroll_shape(coeffs)

    def c2a(cofs):
        array_list = []
        if isinstance(cofs, list) or isinstance(cofs, tuple):
            for c in cofs:
                array_list.append(c2a(c))
            if isinstance(array_list[0], np.ndarray):
                array = np.concatenate(array_list, axis=-1)
            elif isinstance(array_list[0], torch.Tensor):
                array = torch.cat(array_list, -1)
            return array
        elif isinstance(cofs, torch.Tensor) or isinstance(cofs, np.ndarray):
            array = cofs.reshape(cofs.shape[0], -1)
            return array

    coeff_array = c2a(coeffs)
    return coeff_array, shapes

def array_to_coeffs(coeff_array, shapes):

    def a2c(array, sh):
        if (isinstance(sh, tuple) or isinstance(sh, list)) and isinstance(sh[0], int):
            ar = array[...,:np.prod(sh[1:])]
            ar = ar.reshape(sh)
            leftover = array[...,np.prod(sh[1:]):]
            return ar, leftover
        elif (isinstance(sh, tuple) or isinstance(sh, list)) and isinstance(sh[0], tuple):
            leftover = array
            array_tuple = []
            for s in sh:
                ar, leftover = a2c(leftover, s)
                array_tuple.append(ar)
            array_tuple = tuple(array_tuple)
            return array_tuple, leftover
        
    coeffs, _ = a2c(coeff_array, shapes)
    return list(coeffs)

