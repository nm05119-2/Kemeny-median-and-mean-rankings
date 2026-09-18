#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import os
import sys
import re
from collections import defaultdict
from pyscipopt import Model, quicksum


# ============================================================
# LOGGING
# ============================================================

class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.__stdout__

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


# ============================================================
# INPUT
# ============================================================

def load_instance(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    M = len(data)
    N = len(data[0])

    S = []
    for perm in data:
        pos = {perm[i]: i for i in range(N)}
        sij = {}
        for i in range(1, N+1):
            for j in range(1, N+1):
                if i == j:
                    continue
                sij[(i,j)] = 1 if pos[i] < pos[j] else 0
        S.append(sij)

    e = defaultdict(int)
    for s in S:
        for i in range(1, N+1):
            for j in range(i+1, N+1):
                e[(i,j)] += s[(i,j)]

    return N, M, S, e


# ============================================================
# UTILITIES
# ============================================================

def new_model(logname):
    model = Model()
    model.setLogfile(logname)
    model.hideOutput(False)
    return model

def get_full_r(r_half, N):
    """Build full r_{ij} including i>j"""
    r = {}
    for i in range(1, N+1):
        for j in range(1, N+1):
            if i == j:
                continue
            if i < j:
                r[(i,j)] = r_half[(i,j)]
            else:
                r[(i,j)] = 1 - r_half[(j,i)]
    return r


def build_ranking(r_half, N):
    r = get_full_r(r_half, N)

    score = {}
    for i in range(1, N+1):
        score[i] = sum(r[(j,i)] for j in range(1, N+1) if j != i)

    return sorted(score, key=lambda x: score[x])


def add_triangle_constraints(model, r, N):
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            for k in range(j+1, N+1):
                model.addCons(r[(i,j)] + r[(j,k)] - r[(i,k)] >= 0)
                model.addCons(r[(i,j)] + r[(j,k)] - r[(i,k)] <= 1)


# ============================================================
# K-MEDIAN
# ============================================================

def solve_k_median(logname, N, M, e, relax=False):
    model = new_model(logname)

    r = {(i,j): model.addVar(lb=0, ub=1,
                            vtype="C" if relax else "B")
         for i in range(1,N+1)
         for j in range(i+1,N+1)}

    add_triangle_constraints(model, r, N)

    model.setObjective(
        quicksum(r[(i,j)]*(2*e[(i,j)]-M)
        for i in range(1,N+1)
        for j in range(i+1,N+1)),
        "maximize"
    )

    start = time.time()
    model.optimize()
    cpu = time.time() - start

    obj = model.getObjVal()
    nodes = model.getNNodes()

    r_vals = {(i,j): model.getVal(r[(i,j)]) for (i,j) in r}

    return obj, cpu, nodes, r_vals


# ============================================================
# K-MEAN (FULL)
# ============================================================

def solve_k_mean(logname, N, M, S, e, relax=False):
    model = new_model(logname)

    r = {(i,j): model.addVar(lb=0, ub=1,
                            vtype="C" if relax else "B")
         for i in range(1,N+1)
         for j in range(i+1,N+1)}

    theta = {s: model.addVar(lb=0) for s in range(len(S))}

    add_triangle_constraints(model, r, N)

    denom = N*N - N

    for s_idx, sij in enumerate(S):
        for k in range(1, (N*(N-1))//2 + 1):

            model.addCons(
                theta[s_idx] >=
                quicksum(((2*k-1)/denom)*(2*sij[(i,j)]-1)*r[(i,j)]
                for i in range(1,N+1)
                for j in range(i+1,N+1))
                + ((k-k*k)/denom)
                + ((2*k-1)/denom)*sum(1-sij[(i,j)]
                for i in range(1,N+1)
                for j in range(i+1,N+1))
            )

    model.setObjective(
        quicksum(r[(i,j)]*(2*e[(i,j)]-M)
        for i in range(1,N+1)
        for j in range(i+1,N+1))
        - quicksum(theta[s] for s in theta),
        "maximize"
    )

    start = time.time()
    model.optimize()
    cpu = time.time() - start

    obj = model.getObjVal()
    nodes = model.getNNodes()

    r_vals = {(i,j): model.getVal(r[(i,j)]) for (i,j) in r}

    return obj, cpu, nodes, r_vals


# ============================================================
# ROW GENERATION
# ============================================================

def solve_rowgen(logname, N, M, S, e):
    model = new_model(logname)

    r = {(i,j): model.addVar(vtype="B")
         for i in range(1,N+1)
         for j in range(i+1,N+1)}

    theta = {s: model.addVar(lb=0) for s in range(len(S))}

    add_triangle_constraints(model, r, N)

    model.setObjective(
        quicksum(r[(i,j)]*(2*e[(i,j)]-M)
        for i in range(1,N+1)
        for j in range(i+1,N+1))
        - quicksum(theta[s] for s in theta),
        "maximize"
    )

    denom = N*N - N
    total_added = 0
    iterations = 0

    model.setParam("presolving/maxrounds", 0)

    start = time.time()

    while True:
        model.optimize()
        iterations += 1

        r_half = {(i,j): model.getVal(r[(i,j)]) for (i,j) in r}
        r_full = get_full_r(r_half, N)

        violated_cuts=[]

        for s_idx, sij in enumerate(S):

            val = sum((1 - sij[(i,j)]) for i in range(1,N+1) for j in range(i+1,N+1))
            val += sum((2*sij[(i,j)]-1)*r_half[(i,j)]
                       for i in range(1,N+1)
                       for j in range(i+1,N+1))

            rhs = (val*val)/denom
            theta_val = model.getVal(theta[s_idx])

            if theta_val + 1e-6 < rhs:

                k = int(round(val))

                expr = quicksum(((2*k-1)/denom)*(2*sij[(i,j)]-1)*r[(i,j)]
                       for i in range(1,N+1)
                       for j in range(i+1,N+1)) \
                       + ((k-k*k)/denom) \
                       + ((2*k-1)/denom)*sum(1-sij[(i,j)]
                       for i in range(1,N+1)
                       for j in range(i+1,N+1))

                violated_cuts.append((s_idx, expr))

        if not violated_cuts:
            break

        model.freeTransform()

        for s_idx, expr in violated_cuts:
            model.addCons(theta[s_idx] >= expr)

        total_added += len(violated_cuts)

    cpu = time.time() - start

    return model.getObjVal(), model.getNNodes(), iterations, total_added, cpu


# ============================================================
# METRICS
# ============================================================

def compute_metrics(r_half, S, N):
    r = get_full_r(r_half, N)

    phi_L = sum(
        r[(i,j)] * sum(s[(i,j)] for s in S)
        for i in range(1,N+1)
        for j in range(1,N+1)
        if i != j
    )

    theta = 0
    for i in range(1,N+1):
        for j in range(1,N+1):
            if i == j: continue
            for p in range(1,N+1):
                for q in range(1,N+1):
                    if p == q: continue
                    if (i,j) != (p,q):
                        theta += r[(i,j)] * r[(p,q)] * \
                                 sum(s[(i,j)]*s[(p,q)] for s in S)

    phi_Q = (N*N - N - 1)*phi_L - theta

    return phi_L, theta, phi_Q


def weighted_kendall_distance(rank1, rank2, N):
    """
    rank1, rank2: lists like [1,2,3,...]
    returns weighted Kendall distance
    """

    pos1 = {rank1[i]: i+1 for i in range(N)}
    pos2 = {rank2[i]: i+1 for i in range(N)}

    dist = 0.0

    for i in range(1, N+1):
        for j in range(i+1, N+1):

            # check disagreement
            r1 = pos1[i] < pos1[j]
            r2 = pos2[i] < pos2[j]

            if r1 != r2:
                w1 = N - min(pos1[i], pos1[j]) + 1
                w2 = N - min(pos2[i], pos2[j]) + 1
                w = 0.5 * (w1 + w2)

                dist += w

    return dist



def max_weighted_kendall(N):
    # worst case: complete reversal
    maxd = 0
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            w = N - i + 1  # worst case positions
            maxd += w
    return maxd


# ============================================================
# MAIN
# ============================================================

def main(filename):

    base = filename.replace(".json","")
    logname = base + ".log"

    sys.stdout = Tee(logname)
    sys.stderr = sys.stdout

    print("Loading instance...")

    N, M, S, e = load_instance(filename)

    match = re.match(r"n(\d+)m(\d+)i(\d+)", base)
    seed = int(match.group(3)) if match else 0

    # --- MEDIAN ---
    LPm, LPm_cpu, _, _ = solve_k_median(logname,N,M,e,True)
    INTm, INTm_cpu, INTm_nodes, r_m = solve_k_median(logname,N,M,e)

    median_rank = build_ranking(r_m, N)
    gap_median = LPm/INTm - 1

    # --- MEAN ---
    LPmean, LPmean_cpu, _, _ = solve_k_mean(logname,N,M,S,e,True)
    INTmean, INTmean_cpu, INTmean_nodes, r_mean = solve_k_mean(logname,N,M,S,e)

    mean_rank = build_ranking(r_mean, N)
    gap_mean = LPmean/INTmean - 1

    # --- ROWGEN ---
    row_obj, row_nodes, row_iter, row_constr, row_cpu = solve_rowgen(logname,N,M,S,e)

    # --- METRICS ---
    phiL_m, theta_m, phiQ_m = compute_metrics(r_m, S, N)
    phiL_q, theta_q, phiQ_q = compute_metrics(r_mean, S, N)

    actual_error = 1 - (phiQ_m / phiQ_q)
    approx_error = theta_m / ((N*N - N - 1)*phiL_m)
    ratio_error = actual_error / approx_error

    print("Median ranking:", median_rank)
    print("Mean ranking:", mean_rank)
    print("Actual error:", actual_error)

    wk_dist = weighted_kendall_distance(median_rank, mean_rank, N)
    wk_norm = wk_dist / max_weighted_kendall(N)

    print("Weighted Kendall distance:", wk_dist)
    print("Normalized distance:", wk_norm)

    # --- CSV ---
    results_file = "results"

    header = [
        "N","M","seed",
        "LP_median_objval","LP_median_CPU",
        "INT_median_objval","INT_median_nodes","INT_median_CPU",
        "median_ranking","LP_gap_median",
        "LP_mean_objval","LP_mean_CPU",
        "INT_mean_objval","INT_mean_nodes","INT_mean_CPU",
        "mean_ranking","LP_gap_mean",
        "rowgen_objval","rowgen_nodes","rowgen_iterations","rowgen_constrs","rowgen_CPU",
        "actual_error","approx_error","ratio_error",
        "theta_median", "theta_mean", "weighted_Kendall", "normal_Kendall"
    ]

    row = [
        N,M,seed,
        LPm,LPm_cpu,
        INTm,INTm_nodes,INTm_cpu,
        median_rank,gap_median,
        LPmean,LPmean_cpu,
        INTmean,INTmean_nodes,INTmean_cpu,
        mean_rank,gap_mean,
        row_obj,row_nodes,row_iter,row_constr,row_cpu,
        actual_error,approx_error,ratio_error,
        theta_m, theta_q, wk_dist, wk_norm
    ]

    if not os.path.exists(results_file):
        with open(results_file,"w") as f:
            f.write(";".join(header) + "\n")

    with open(results_file,"a") as f:
        f.write(";".join(map(str,row)) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py nNmMiX.json")
        sys.exit(1)

    main(sys.argv[1])
