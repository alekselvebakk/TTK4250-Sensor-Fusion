from typing import Tuple

import numpy as np


def discrete_bayes(
    # the prior: shape=(n,)
    # p(imm)
    pr: np.ndarray,
    # the conditional/likelihood: shape=(n, m) 
    # p()
    cond_pr: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray
]:  # the new marginal and conditional: shapes=((m,), (m, n))
    """
    Swap which discrete variable is the marginal and conditional.
    We want to update our belief of hypothesis x,given evidence z
    We have p(x) and p(z|x)
    we get p(x|z)

    """


    joint = cond_pr@np.diag(pr)# TODO

    marginal = joint.sum(axis=1)# TODO

    # Take care of rare cases of degenerate zero marginal,
    conditional = np.divide(joint, 
                            marginal[:, None], 
                            where = marginal[None] > 0) # TODO

    # flip axes?? (n, m) -> (m, n)
    # conditional = conditional.T

    # optional DEBUG
    assert np.all(
        np.isfinite(conditional)
    ), f"NaN or inf in conditional in discrete bayes"
    assert np.all(
        np.less_equal(0, conditional)
    ), f"Negative values for conditional in discrete bayes"
    assert np.all(
        np.less_equal(conditional, 1)
    ), f"Value more than on in discrete bayes"

    assert np.all(np.isfinite(marginal)), f"NaN or inf in marginal in discrete bayes"

    return marginal, conditional
