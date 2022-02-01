from itm.distributions import marginal
from itm.measures import entropy
from itm.measures import conditional_entropy

def mutual_information(prob_xy):
    ''' Mutual Information
        I(X,Y) = H(X) - H(X|Y)
        Assumes there are two columns in prob_xy'''
    prob_x = marginal(prob_xy, 0)
    return entropy(prob_x) - conditional_entropy(prob_xy, 0)

def mutual_information2(prob_xy):
    ''' Mutual Information
        I(X,Y) = H(X, Y) - H(X|Y) - H(Y|X)
        Assumes there are two columns in prob_xy'''
    return  entropy(prob_xy)- conditional_entropy(prob_xy, 0)- conditional_entropy(prob_xy, 1)

def mutual_information3(prob_xy):
    ''' Mutual Information
        I(X,Y) = H(X) + H(Y) - H(X,Y)
        Assumes there are two columns in prob_xy'''
    prob_x = marginal(prob_xy, 0)
    prob_y = marginal(prob_xy, 1)
    return entropy(prob_x) + entropy(prob_y) - entropy(prob_xy)

def conditional_mutual_information(prob_xyz, *cols):
    ''' Conditional Mutual Information
        I(X,Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
        Assumes, after conditioning on cols, there are two columns in prob_XYZ'''
    xy_cols = [c for c in range(len(cols)+2) if c not in cols]
    x_col = xy_cols[0]
    y_col = xy_cols[1]
    cols_and_x = [c for c in cols if c < x_col] + [x_col] + [c for c in cols if c > x_col]
    cols_and_y = [c for c in cols if c < y_col] + [y_col] + [c for c in cols if c > y_col]

    prob_x_z = marginal(prob_xyz, *cols_and_x)
    prob_y_z = marginal(prob_xyz, *cols_and_y)
    prob_z = marginal(prob_xyz, *cols)

    return (
        entropy(prob_x_z) + entropy(prob_y_z) - entropy(prob_xyz) - entropy(prob_z)
    )

def conditional_mutual_information2(prob_xyz, *cols):
    ''' Conditional Mutual Information
        I(X,Y|Z) = H(X |Z) - H(X|Y, Z)
        Assumes, after conditioning on cols, there are two columns in prob_xyz'''
    xy_cols = [c for c in range(len(cols)+2) if c not in cols]
    x_col = xy_cols[0]
    y_col = xy_cols[1]
    cols_and_x = [c for c in cols if c < x_col] + [x_col] + [c for c in cols if c > x_col]
    cols_and_y = [c for c in cols if c < y_col] + [y_col] + [c for c in cols if c > y_col]
    cols_shifted = [ c for c in cols if c < y_col] + [c-1 for c in cols if c > y_col]
    prob_xz = marginal(prob_xyz, *cols_and_x)
    return (
        conditional_entropy(prob_xz, *cols_shifted)
        - conditional_entropy(prob_xyz, *cols_and_y)
    )

def conditional_mutual_information3(prob_xyz, *cols):
    ''' Conditional Mutual Information
        I(X,Y|Z) = H(X |Z) + H(Y |Z) - H(X, Y|Z)
        Assumes, after conditioning on cols, there are two columns in prob_xyz'''
    xy_cols = [c for c in range(len(cols)+2) if c not in cols]
    x_col = xy_cols[0]
    y_col = xy_cols[1]
    cols_and_x = [c for c in cols if c < x_col] + [x_col] + [c for c in cols if c > x_col]
    cols_and_y = [c for c in cols if c < y_col] + [y_col] + [c for c in cols if c > y_col]
    cols_shifted_by_x = [
        c for c in cols if c < x_col] + [c-1 for c in cols if c > x_col]
    cols_shifted_by_y = [
        c for c in cols if c < y_col] + [c-1 for c in cols if c > y_col]
    prob_yz = marginal(prob_xyz, *cols_and_y)
    prob_xz = marginal(prob_xyz, *cols_and_x)
    return (
          conditional_entropy(prob_yz, *cols_shifted_by_x)
        + conditional_entropy(prob_xz, *cols_shifted_by_y)
        - conditional_entropy(prob_xyz, *cols)
    )
