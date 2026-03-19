from snowflake.snowpark import functions as F


def stratified_sample_snowpark(df, label_col: str, neg_frac: float, pos_frac: float, max_neg: int = None, max_pos: int = None):
    """
    Sample by class in Snowpark; cap each class so union does not drop a class.
    Note: Snowpark sample() has no seed, so sampling may vary run-to-run.
    """
    neg = df.filter(F.col(label_col) == 0).sample(neg_frac)
    pos = df.filter(F.col(label_col) == 1).sample(pos_frac)
    if max_neg is not None:
        neg = neg.limit(max_neg)
    if max_pos is not None:
        pos = pos.limit(max_pos)
    return neg.union_all(pos)

