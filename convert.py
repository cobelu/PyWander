# Connor Luckett
# Wander
# Converts a file of ratings into a pickled sparse matrix

import pickle
from scipy.sparse import save_npz, csr_matrix, coo_matrix
import pandas as pd
import numpy as np
import os


def main():
    in_file = "/Users/cobelu/Documents/Research/mf/data/ml-25m/train.txt"
    out_file = "data/movielens.npz"

    # Load the file
    df = pd.read_csv(in_file, header=None, sep='\t')
    print(df.head())
    print("Shape:", df.shape)
    # Make sure there are NOT negative entries
    df = df[(df >= 0).all(1)]
    # Select the columns
    user_col = df[0]
    item_col = df[1]
    rating_col = df[2]
    print(df.head())
    # Get the columns as numpy arrays
    users = user_col.to_numpy()
    items = item_col.to_numpy()
    ratings = rating_col.to_numpy()

    print("Users:", user_col.unique())
    print("Items:", item_col.unique())

    # Get the dimensions of the sparse matrix
    num_users = user_col.max()
    num_items = item_col.max()

    print("Num users:", num_users)
    print("Num items:", num_items)

    # Create the sparse matrix as a CSR
    sparse_matrix: csr_matrix = coo_matrix((ratings, (users, items)), dtype=np.float64).tocsr()
    # Save the sparse matrix
    # if not (os.path.exists(out_file)):
    #     Create the directory if needed
    #     os.mkdir(out_file)
    save_npz(out_file, sparse_matrix)


if __name__ == '__main__':
    main()
