#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# clever.py
#
# Compute CLEVER score using collected Lipschitz constants
#
# Copyright (C) 2017-2018, IBM Corp.
# Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
#                 and Huan Zhang <ecezhang@ucdavis.edu>
#
# This program is licenced under the Apache 2.0 licence,
# contained in the LICENCE file in this directory.

import os
import sys
import glob
import scipy
import scipy.optimize
import argparse
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from functools import partial
from multiprocessing import Pool
from scipy.stats import weibull_min


# We observe that the scipy.optimize.fmin optimizer (using Nelder–Mead method) sometimes diverges to very large
# parameters a, b and c. Thus, we add a very small regularization to the MLE optimization process to avoid this
# divergence
def fmin_with_reg(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None, maxfun=None, full_output=0, disp=1, retall=0,
                  callback=None, initial_simplex=None, shape_reg=0.01):
    def func_with_reg(theta, x):
        shape = theta[2]
        log_likelyhood = func(theta, x)
        reg = shape_reg * shape * shape
        # penalize the shape parameter
        return log_likelyhood + reg

    return scipy.optimize.fmin(func_with_reg, x0, args, xtol, ftol, maxiter, maxfun, full_output, disp, retall,
                               callback, initial_simplex)


# fit using weibull_min.fit and run a K-S test
def fit_and_test(rescaled_sample, sample, loc_shift, shape_rescale, optimizer, c_i):
    [c, loc, scale] = weibull_min.fit(-rescaled_sample, c_i, optimizer=optimizer)
    loc = - loc_shift + loc * shape_rescale
    scale *= shape_rescale
    ks, pVal = scipy.stats.kstest(-sample, 'weibull_min', args=(c, loc, scale))
    return c, loc, scale, ks, pVal


def plot_weibull(sample, c, loc, scale, ks, pVal, p, q, figname):
    # compare the sample histogram and fitting result
    fig, ax = plt.subplots(1, 1)

    x = np.linspace(-1.01 * max(sample), -0.99 * min(sample), 100)
    ax.plot(x, weibull_min.pdf(x, c, loc, scale), 'r-', label='fitted pdf ' + p + '-bnd')
    ax.hist(-sample, normed=True, bins=20, histtype='stepfilled')
    ax.legend(loc='best', frameon=False)
    plt.xlabel('-Lips_' + q)
    plt.ylabel('pdf')
    plt.title('c = {:.2f}, loc = {:.2f}, scale = {:.2f}, ks = {:.2f}, pVal = {:.2f}'.format(c, loc, scale, ks, pVal))
    plt.savefig(figname)
    plt.close()


# We observe than the MLE estimator in scipy sometimes can converge to a bad value if the inital shape parameter c is
# too far from the true value. Thus we test a few different initializations and choose the one with best p-value all
# the initializations are tested in parallel; remove some of them to speedup computation.
c_init = [0.1, 1, 5, 10, 20, 50, 100]


def get_best_weibull_fit(sample, use_reg=False, shape_reg=0.01):
    # initialize dictionary to save the fitting results
    fitted_paras = {"c": [], "loc": [], "scale": [], "ks": [], "pVal": []}
    # reshape the data into a better range, this helps the MLE solver find the solution easier
    loc_shift = np.amax(sample)
    dist_range = np.amax(sample) - np.amin(sample)
    shape_rescale = dist_range
    print("shape rescale = {}".format(shape_rescale))
    rescaled_sample = np.copy(sample)
    rescaled_sample -= loc_shift
    rescaled_sample /= shape_rescale
    print("loc_shift = {}".format(loc_shift))

    # fit weibull distn: sample follows reverse weibull dist, so -sample follows weibull distribution
    if use_reg:
        results = pool.map(partial(fit_and_test, rescaled_sample, sample, loc_shift, shape_rescale,
                                   partial(fmin_with_reg, shape_reg=shape_reg)), c_init)
    else:
        results = pool.map(partial(fit_and_test, rescaled_sample, sample, loc_shift, shape_rescale,
                                   scipy.optimize.fmin), c_init)

    for res, c_i in zip(results, c_init):
        c = res[0]
        loc = res[1]
        scale = res[2]
        ks = res[3]
        pVal = res[4]
        print("[DEBUG][L2] c_init = {:5.5g}, fitted c = {:6.2f}, loc = {:7.2f}, scale = {:7.2f}, ks = {:4.2f}, "
              "pVal = {:4.2f}, max = {:7.2f}".format(c_i, c, loc, scale, ks, pVal, loc_shift))

        fitted_paras['c'].append(c)
        fitted_paras['loc'].append(loc)
        fitted_paras['scale'].append(scale)
        fitted_paras['ks'].append(ks)
        fitted_paras['pVal'].append(pVal)

    # get the paras of best pVal among c_init
    max_pVal = np.nanmax(fitted_paras['pVal'])
    if np.isnan(max_pVal) or max_pVal < 0.001:
        print("ill-conditioned samples. Using maximum sample value.")
        # handle the ill conditioned case
        return -1, -1, -max(sample), -1, -1, -1

    max_pVal_idx = fitted_paras['pVal'].index(max_pVal)

    c_init_best = c_init[max_pVal_idx]
    c_best = fitted_paras['c'][max_pVal_idx]
    loc_best = fitted_paras['loc'][max_pVal_idx]
    scale_best = fitted_paras['scale'][max_pVal_idx]
    ks_best = fitted_paras['ks'][max_pVal_idx]
    pVal_best = fitted_paras['pVal'][max_pVal_idx]

    return c_init_best, c_best, loc_best, scale_best, ks_best, pVal_best


def get_lipschitz_estimate(G_max, norm="L2", use_reg=False, shape_reg=0.01):  # G_max = array of max values
    global plot_res
    c_init, c, loc, scale, ks, pVal = get_best_weibull_fit(G_max, use_reg, shape_reg)

    # the norm here is Lipschitz constant norm, not the bound's norm
    if norm == "L1":
        p = "i"
        q = "1"
    elif norm == "L2":
        p = "2"
        q = "2"
    elif norm == "Li":
        p = "1"
        q = "i"
    else:
        raise RuntimeError("Lipschitz norm is not in 1, 2, i!")

    figname = "_L%s.png" % p

    if plot_res is not None:
        plot_res.get()

    if figname:
        plot_res = pool.apply_async(plot_weibull, (G_max, c, loc, scale, ks, pVal, p, q, figname))

    # Return the Weibull position parameter
    return {'Lips_est': -loc, 'shape': c, 'loc': loc, 'scale': scale, 'ks': ks, 'pVal': pVal}


# file name contains some information, like sample_id, true_label and target_label
def parse_filename(filename):
    basename = os.path.basename(filename)
    name, _ = os.path.splitext(basename)
    name_arr = name.split('_')
    Nsamp = int(name_arr[0])
    Niters = int(name_arr[1])
    sample_id = int(name_arr[2])
    true_label = int(name_arr[3])
    target_label = int(name_arr[4])
    image_info = name_arr[5]
    activation = name_arr[6]
    return Nsamp, Niters, sample_id, true_label, target_label, image_info, activation


if __name__ == "__main__":
    # parse command line parameters
    parser = argparse.ArgumentParser(description='Compute CLEVER scores using collected gradient norm data.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_folder',
                        help='data folder path')
    parser.add_argument('--min', dest='reduce_op', action='store_const',
                        default=lambda x: sum(x) / float(len(x)) if len(x) > 0 else 0, const=min,
                        help='report min of all CLEVER scores instead of avg')
    parser.add_argument('--user_type', default="",
                        help='replace user type with string, used for ImageNet data processing')
    parser.add_argument('--use_slope', action="store_true",
                        help='report slope estimate. To use this option, collect_gradients.py needs to be run '
                             'with --compute_slope')
    parser.add_argument('--untargeted', action="store_true",
                        help='process untargeted attack results (for MNIST and CIFAR)')
    parser.add_argument('--num_samples', type=int, default=0,
                        help='the number of samples to use. Default 0 is to use all samples')
    parser.add_argument('--num_images', type=int, default=0,
                        help='number of images to use, 0 to use all images')
    parser.add_argument('--shape_reg', default=0.01, type=float,
                        help='to avoid the MLE solver in Scipy to diverge, we add a small regularization '
                             '(default 0.01 is sufficient)')
    parser.add_argument('--nthreads', default=0, type=int,
                        help='number of threads (default is len(c_init)+1)')
    parser.add_argument('--plot_dir', default='',
                        help='output path for weibull fit figures (empty to disable)')
    parser.add_argument('--method', default="mle_reg", choices=['mle', 'mle_reg', 'maxsamp'],
                        help='Fitting algorithm. Please use mle_reg for best results')
    args = vars(parser.parse_args())
    reduce_op = args['reduce_op']
    if args['plot_dir']:
        os.system("mkdir -p " + args['plot_dir'])
    print(args)

    # create thread pool
    if args['nthreads'] == 0:
        args['nthreads'] = len(c_init) + 1
    print("using {} threads".format(args['nthreads']))
    pool = Pool(processes=args['nthreads'])
    # used for asynchronous plotting in background
    plot_res = None

    # get a list of all '.mat' files in folder
    file_list = glob.glob(args['data_folder'] + '/*.mat')

    # sort by image ID, then by information (least likely, random, top-2)
    file_list = sorted(file_list, key=lambda x: (parse_filename(x)[2], parse_filename(x)[5]))

    # get the first num_images files
    if args['num_images']:
        file_list = file_list[:args['num_images']]

    if args['untargeted']:
        bounds = {}  # bounds will be inserted per image
    else:
        # aggregate information for three different types: least, random and top2
        # each has three bounds: L1, L2, and Linf
        bounds = {"least": [[], [], []],
                  "random": [[], [], []],
                  "top2": [[], [], []]}

    for fname in file_list:
        nsamps, niters, sample_id, true_label, target_label, img_info, activation = parse_filename(fname)

        # keys in mat:
        # ['Li_max', 'pred', 'G1_max', 'g_x0', 'path', 'info', 'G2_max', 'true_label', 'args', 'L1_max', 'Gi_max',
        #  'L2_max', 'id', 'target_label']
        mat = sio.loadmat(fname)
        print('loading {}'.format(fname))

        if args['use_slope']:
            G1_max = np.squeeze(mat['L1_max'])
            G2_max = np.squeeze(mat['L2_max'])
            Gi_max = np.squeeze(mat['Li_max'])
        else:
            G1_max = np.squeeze(mat['G1_max'])
            G2_max = np.squeeze(mat['G2_max'])
            Gi_max = np.squeeze(mat['Gi_max'])

        if args['num_samples'] != 0:
            prev_len = len(G1_max)
            G1_max = G1_max[:args['num_samples']]
            G2_max = G2_max[:args['num_samples']]
            Gi_max = Gi_max[:args['num_samples']]
            print('Using {} out of {} total samples'.format(len(G1_max), prev_len))

        g_x0 = np.squeeze(mat['g_x0'])
        sample_id = np.squeeze(mat['id'])
        target_label = np.squeeze(mat['target_label'])
        true_label = np.squeeze(mat['true_label'])
        img_info = mat['info'][0]
        if args['user_type'] != "" and img_info == "user":
            img_info = args['user_type']

        if args['method'] == "maxsamp":
            Est_G1 = {'Lips_est': max(G1_max), 'shape': -1, 'loc': -1, 'scale': -1, 'ks': -1, 'pVal': -1}
            Est_G2 = {'Lips_est': max(G2_max), 'shape': -1, 'loc': -1, 'scale': -1, 'ks': -1, 'pVal': -1}
            Est_Gi = {'Lips_est': max(Gi_max), 'shape': -1, 'loc': -1, 'scale': -1, 'ks': -1, 'pVal': -1}

        elif args['method'] == "mle":
            # estimate Lipschitz constant: Est_G1 is a dictionary containing Lips_est and weibull paras        
            Est_G1 = get_lipschitz_estimate(G1_max, "L1")
            Est_G2 = get_lipschitz_estimate(G2_max, "L2")
            Est_Gi = get_lipschitz_estimate(Gi_max, "Li")

        elif args['method'] == "mle_reg":
            print('estimating L1...')
            Est_G1 = get_lipschitz_estimate(G1_max, "L1", True, args['shape_reg'])
            print('estimating L2...')
            Est_G2 = get_lipschitz_estimate(G2_max, "L2", True, args['shape_reg'])
            print('estimating Li...')
            Est_Gi = get_lipschitz_estimate(Gi_max, "Li", True, args['shape_reg'])

        else:
            raise RuntimeError("method not supported")

        # the estimated Lipschitz constant
        Lip_G1 = Est_G1['Lips_est']
        Lip_G2 = Est_G2['Lips_est']
        Lip_Gi = Est_Gi['Lips_est']

        # the estimated shape parameter (c) in Weibull distn 
        shape_G1 = Est_G1['shape']
        shape_G2 = Est_G2['shape']
        shape_Gi = Est_Gi['shape']

        # the estimated loc parameters in Weibull distn
        loc_G1 = Est_G1['loc']
        loc_G2 = Est_G2['loc']
        loc_Gi = Est_Gi['loc']

        # the estimated scale parameters in Weibull distn
        scale_G1 = Est_G1['scale']
        scale_G2 = Est_G2['scale']
        scale_Gi = Est_Gi['scale']

        # the computed ks score
        ks_G1 = Est_G1['ks']
        ks_G2 = Est_G2['ks']
        ks_Gi = Est_Gi['ks']

        # the computed pVal
        pVal_G1 = Est_G1['pVal']
        pVal_G2 = Est_G2['pVal']
        pVal_Gi = Est_Gi['pVal']

        # compute robustness bound
        bnd_L1 = g_x0 / Lip_Gi
        bnd_L2 = g_x0 / Lip_G2
        bnd_Li = g_x0 / Lip_G1

        # save bound of each image
        if args['untargeted']:
            sample_id = int(sample_id)
            if sample_id not in bounds:
                bounds[sample_id] = [[], [], []]
            bounds[sample_id][0].append(bnd_L1)
            bounds[sample_id][1].append(bnd_L2)
            bounds[sample_id][2].append(bnd_Li)
        else:
            bounds[img_info][0].append(bnd_L1)
            bounds[img_info][1].append(bnd_L2)
            bounds[img_info][2].append(bnd_Li)

        # original data_process mode
        bndnorm_L1 = "1"
        bndnorm_L2 = "2"
        bndnorm_Li = "i"

        if args['method'] == "maxsamp":
            print('[DEBUG][L1] id = {}, true_label = {}, target_label = {}, info = {}, nsamps = {}, niters = {}, '
                  'bnd_norm = {}, bnd = {:.5g}'.format(sample_id, true_label, target_label, img_info, nsamps, niters,
                                                       bndnorm_L1, bnd_L1))
            print('[DEBUG][L1] id = {}, true_label = {}, target_label = {}, info = {}, nsamps = {}, niters = {}, '
                  'bnd_norm = {}, bnd = {:.5g}'.format(sample_id, true_label, target_label, img_info, nsamps, niters,
                                                       bndnorm_L2, bnd_L2))
            print('[DEBUG][L1] id = {}, true_label = {}, target_label = {}, info = {}, nsamps = {}, niters = {}, '
                  'bnd_norm = {}, bnd = {:.5g}'.format(sample_id, true_label, target_label, img_info, nsamps, niters,
                                                       bndnorm_Li, bnd_Li))

        elif args['method'] == "mle" or args['method'] == "mle_reg":
            # estimate Lipschitz constant: Est_G1 is a dictionary containing Lips_est and weibull paras
            # current debug mode: bound_L1 corresponds to Gi, bound_L2 corresponds to G2, bound_Li corresponds to G1
            print('[DEBUG][L1] id = {}, true_label = {}, target_label = {}, info = {}, nsamps = {}, niters = {}, '
                  'bnd_norm = {}, bnd = {:.5g}, ks = {:.5g}, pVal = {:.5g}, shape = {:.5g}, loc = {:.5g}, '
                  'scale = {:.5g}, g_x0 = {}'.format(sample_id, true_label, target_label, img_info, nsamps, niters,
                                                     bndnorm_L1, bnd_L1, ks_Gi, pVal_Gi, shape_Gi, loc_Gi, scale_Gi,
                                                     g_x0))
            print('[DEBUG][L1] id = {}, true_label = {}, target_label = {}, info = {}, nsamps = {}, niters = {}, '
                  'bnd_norm = {}, bnd = {:.5g}, ks = {:.5g}, pVal = {:.5g}, shape = {:.5g}, loc = {:.5g}, '
                  'scale = {:.5g}, g_x0 = {}'.format(sample_id, true_label, target_label, img_info, nsamps, niters,
                                                     bndnorm_L2, bnd_L2, ks_G2, pVal_G2, shape_G2, loc_G2, scale_G2,
                                                     g_x0))
            print('[DEBUG][L1] id = {}, true_label = {}, target_label = {}, info = {}, nsamps = {}, niters = {}, '
                  'bnd_norm = {}, bnd = {:.5g}, ks = {:.5g}, pVal = {:.5g}, shape = {:.5g}, loc = {:.5g}, '
                  'scale = {:.5g}, g_x0 = {}'.format(sample_id, true_label, target_label, img_info, nsamps, niters,
                                                     bndnorm_Li, bnd_Li, ks_G1, pVal_G1, shape_G1, loc_G1, scale_G1,
                                                     g_x0))

        else:
            raise RuntimeError("method not supported")

        sys.stdout.flush()

    if args['untargeted']:
        clever_L1s = []
        clever_L2s = []
        clever_Lis = []
        for sample_id, sample_bounds in bounds.items():
            img_clever_L1 = min(sample_bounds[0])
            img_clever_L2 = min(sample_bounds[1])
            img_clever_Li = min(sample_bounds[2])
            n_classes = len(sample_bounds[0]) + 1
            assert len(sample_bounds[0]) == len(sample_bounds[2])
            assert len(sample_bounds[1]) == len(sample_bounds[2])
            print('[STATS][L1] image = {:3d}, n_classes = {:3d}, clever_L1 = {:.5g}, clever_L2 = {:.5g}, '
                  'clever_Li = {:.5g}'.format(sample_id, n_classes, img_clever_L1, img_clever_L2, img_clever_Li))
            clever_L1s.append(img_clever_L1)
            clever_L2s.append(img_clever_L2)
            clever_Lis.append(img_clever_Li)
        info = "untargeted"
        clever_L1 = reduce_op(clever_L1s)
        clever_L2 = reduce_op(clever_L2s)
        clever_Li = reduce_op(clever_Lis)
        print('[STATS][L0] info = {}, '
              '{}_clever_L1 = {:.5g}, {}_clever_L2 = {:.5g}, {}_clever_Li = {:.5g}'.format(info, info, clever_L1, info,
                                                                                           clever_L2, info, clever_Li))
    else:
        # print min/average bound
        for info, info_bounds in bounds.items():
            # reduce each array to a single number (min or avg)
            clever_L1 = reduce_op(info_bounds[0])
            clever_L2 = reduce_op(info_bounds[1])
            clever_Li = reduce_op(info_bounds[2])
            print('[STATS][L0] info = {}, {}_clever_L1 = {:.5g}, {}_clever_L2 = {:.5g}, {}_clever_Li = {:.5g}'.format(
                    info, info, clever_L1, info, clever_L2, info, clever_Li))
            sys.stdout.flush()

    # shutdown thread pool
    pool.close()
    pool.join()
