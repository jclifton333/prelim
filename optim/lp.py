import numpy as np
import docplex.mp
import docplex.mp.model as dcpm


def lp_max(coef, intercept, budget):
    """
    Solve min coef.A + intercept, subject to constraints

    :param coef:
    :param intercept:
    :param budget:
    :return:
    """
    L = len(coef)
    model = dcpm.Model(name="lp_max")
    vars = {i: model.binary_var(name="trt_{}".format(i)) for i in range(L)}
    obj = model.sum(vars[i]*float(coef[i]) for i in range(L)) + float(intercept)
    model.add_constraint(model.sum(vars[i] for i in range(L)) == budget)
    model.minimize(obj)
    sol = model.solve(url=None, key=None)
    argmax = np.array([int(sol.get_value(f'trt_{i}')) for i in range(L)])
    return argmax