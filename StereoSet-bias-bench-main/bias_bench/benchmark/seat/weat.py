import itertools
import math

import numpy as np
import scipy.special
import scipy.stats

# X and Y are two sets of target words of equal size.
# A and B are two sets of attribute words.


def cossim(x, y):
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))


def construct_cossim_lookup(XY, AB):
    """Args:
        XY: Mapping from target string to target vector (either in X or Y).
        AB: Mapping from attribute string to attribute vectore (either in A or B).

    Returns:
        An array of size (len(XY), len(AB)) containing cosine similarities
        between items in XY and items in AB.
    """
    cossims = np.zeros((len(XY), len(AB)))
    for xy in XY:
        for ab in AB:
            cossims[xy, ab] = cossim(XY[xy], AB[ab])
    return cossims


def s_wAB(A, B, cossims):
    """Returns:
    Vector of s(w, A, B) across w, where
        s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
    """
    return cossims[:, A].mean(axis=1) - cossims[:, B].mean(axis=1)


def s_XAB(X, s_wAB_memo):
    r"""Given indices of target concept X and precomputed s_wAB values,
    return slightly more computationally efficient version of WEAT
    statistic for p-value computation.

    Caliskan defines the WEAT statistic s(X, Y, A, B) as
        sum_{x in X} s(x, A, B) - sum_{y in Y} s(y, A, B)
    where s(w, A, B) is defined as
        mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
    The p-value is computed using a permutation test on (X, Y) over all
    partitions (X', Y') of X union Y with |X'| = |Y'|.

    However, for all partitions (X', Y') of X union Y,
        s(X', Y', A, B)
      = sum_{x in X'} s(x, A, B) + sum_{y in Y'} s(y, A, B)
      = C,
    a constant.  Thus
        sum_{x in X'} s(x, A, B) + sum_{y in Y'} s(y, A, B)
      = sum_{x in X'} s(x, A, B) + (C - sum_{x in X'} s(x, A, B))
      = C + 2 sum_{x in X'} s(x, A, B).

    By monotonicity,
        s(X', Y', A, B) > s(X, Y, A, B)
    if and only if
        [s(X', Y', A, B) - C] / 2 > [s(X, Y, A, B) - C] / 2,
    that is,
        sum_{x in X'} s(x, A, B) > sum_{x in X} s(x, A, B).
    Thus we only need use the first component of s(X, Y, A, B) as our
    test statistic.
    """
    return s_wAB_memo[X].sum()


def s_XYAB(X, Y, s_wAB_memo):
    r"""Given indices of target concept X and precomputed s_wAB values,
    the WEAT test statistic for p-value computation.
    """
    return s_XAB(X, s_wAB_memo) - s_XAB(Y, s_wAB_memo)


def p_val_permutation_test(X, Y, A, B, n_samples, cossims, parametric=False):
    """Compute the p-val for the permutation test, which is defined as
    the probability that a random even partition X_i, Y_i of X u Y
    satisfies P[s(X_i, Y_i, A, B) > s(X, Y, A, B)]
    """
    X = np.array(list(X), dtype=np.int)
    Y = np.array(list(Y), dtype=np.int)
    A = np.array(list(A), dtype=np.int)
    B = np.array(list(B), dtype=np.int)

    assert len(X) == len(Y)
    size = len(X)
    s_wAB_memo = s_wAB(A, B, cossims=cossims)
    XY = np.concatenate((X, Y))

    if parametric:
        print("Using parametric test")
        s = s_XYAB(X, Y, s_wAB_memo)

        print("Drawing {} samples".format(n_samples))
        samples = []
        for _ in range(n_samples):
            np.random.shuffle(XY)
            Xi = XY[:size]
            Yi = XY[size:]
            assert len(Xi) == len(Yi)
            si = s_XYAB(Xi, Yi, s_wAB_memo)
            samples.append(si)

        # Compute sample standard deviation and compute p-value by
        # assuming normality of null distribution
        print("Inferring p-value based on normal distribution")
        (shapiro_test_stat, shapiro_p_val) = scipy.stats.shapiro(samples)
        print(
            "Shapiro-Wilk normality test statistic: {:.2g}, p-value: {:.2g}".format(
                shapiro_test_stat, shapiro_p_val
            )
        )
        sample_mean = np.mean(samples)
        sample_std = np.std(samples, ddof=1)
        print(
            "Sample mean: {:.2g}, sample standard deviation: {:.2g}".format(
                sample_mean, sample_std
            )
        )
        p_val = scipy.stats.norm.sf(s, loc=sample_mean, scale=sample_std)
        return p_val

    else:
        print("Using non-parametric test")
        s = s_XAB(X, s_wAB_memo)
        total_true = 0
        total_equal = 0
        total = 0

        num_partitions = int(scipy.special.binom(2 * len(X), len(X)))
        if num_partitions > n_samples:
            # We only have as much precision as the number of samples drawn;
            # bias the p-value (hallucinate a positive observation) to
            # reflect that.
            total_true += 1
            total += 1
            print("Drawing {} samples (and biasing by 1)".format(n_samples - total))
            for _ in range(n_samples - 1):
                np.random.shuffle(XY)
                Xi = XY[:size]
                assert 2 * len(Xi) == len(XY)
                si = s_XAB(Xi, s_wAB_memo)
                if si > s:
                    total_true += 1
                elif si == s:  # use conservative test
                    total_true += 1
                    total_equal += 1
                total += 1

        else:
            print("Using exact test ({} partitions)".format(num_partitions))
            for Xi in itertools.combinations(XY, len(X)):
                Xi = np.array(Xi, dtype=np.int)
                assert 2 * len(Xi) == len(XY)
                si = s_XAB(Xi, s_wAB_memo)
                if si > s:
                    total_true += 1
                elif si == s:  # use conservative test
                    total_true += 1
                    total_equal += 1
                total += 1

        if total_equal:
            print("Equalities contributed {}/{} to p-value".format(total_equal, total))

        return total_true / total


def mean_s_wAB(X, A, B, cossims):
    return np.mean(s_wAB(A, B, cossims[X]))


def stdev_s_wAB(X, A, B, cossims):
    return np.std(s_wAB(A, B, cossims[X]), ddof=1)


def effect_size(X, Y, A, B, cossims):
    """Compute the effect size, which is defined as
        [mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B)] /
            [ stddev_{w in X u Y} s(w, A, B) ]
    Args:
        X, Y, A, B : sets of target (X, Y) and attribute (A, B) indices
    """
    X = list(X)
    Y = list(Y)
    A = list(A)
    B = list(B)

    numerator = mean_s_wAB(X, A, B, cossims=cossims) - mean_s_wAB(
        Y, A, B, cossims=cossims
    )
    denominator = stdev_s_wAB(X + Y, A, B, cossims=cossims)
    return numerator / denominator


def convert_keys_to_ints(X, Y):
    return (
        dict((i, v) for (i, (k, v)) in enumerate(X.items())),
        dict((i + len(X), v) for (i, (k, v)) in enumerate(Y.items())),
    )


def run_test(encs, n_samples, parametric=False):
    """Run a WEAT.
    Args:
        encs (Dict[str: Dict]): dictionary mapping targ1, targ2, attr1, attr2
            to dictionaries containing the category and the encodings
        n_samples (int): number of samples to draw to estimate p-value
            (use exact test if number of permutations is less than or
            equal to n_samples)
    """
    X, Y = encs["targ1"]["encs"], encs["targ2"]["encs"]
    A, B = encs["attr1"]["encs"], encs["attr2"]["encs"]

    # First convert all keys to ints to facilitate array lookups
    (X, Y) = convert_keys_to_ints(X, Y)
    (A, B) = convert_keys_to_ints(A, B)

    XY = X.copy()
    XY.update(Y)
    AB = A.copy()
    AB.update(B)

    print("Computing cosine similarities...")
    cossims = construct_cossim_lookup(XY, AB)

    print(
        "Null hypothesis: no difference between {} and {} in association to attributes {} and {}".format(
            encs["targ1"]["category"],
            encs["targ2"]["category"],
            encs["attr1"]["category"],
            encs["attr2"]["category"],
        )
    )
    print("Computing pval...")
    pval = p_val_permutation_test(
        X, Y, A, B, n_samples, cossims=cossims, parametric=parametric
    )
    print(f"pval: {pval:.3f}")

    print("computing effect size...")
    esize = effect_size(X, Y, A, B, cossims=cossims)
    print(f"esize: {esize:.3f}")
    return esize, pval


if __name__ == "__main__":
    X = {"x" + str(i): 2 * np.random.rand(10) - 1 for i in range(25)}
    Y = {"y" + str(i): 2 * np.random.rand(10) - 1 for i in range(25)}
    A = {"a" + str(i): 2 * np.random.rand(10) - 1 for i in range(25)}
    B = {"b" + str(i): 2 * np.random.rand(10) - 1 for i in range(25)}
    A = X
    B = Y

    (X, Y) = convert_keys_to_ints(X, Y)
    (A, B) = convert_keys_to_ints(A, B)

    XY = X.copy()
    XY.update(Y)
    AB = A.copy()
    AB.update(B)

    cossims = construct_cossim_lookup(XY, AB)
    print("computing pval...")
    pval = p_val_permutation_test(X, Y, A, B, cossims=cossims, n_samples=10000)
    print("pval: %g".format(pval))

    print("computing effect size...")
    esize = effect_size(X, Y, A, B, cossims=cossims)
    print("esize: %g".format(esize))
