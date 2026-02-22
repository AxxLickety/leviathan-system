def split_train_test(df, train_end="2018-12-31"):
    """
    Split dataframe into train/test by date.
    All rows with date <= train_end go to train.
    """
    df = df.copy()
    train = df[df["date"] <= train_end]
    test  = df[df["date"] > train_end]
    return train, test

