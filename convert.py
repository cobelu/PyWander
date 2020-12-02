# Connor Luckett
# Wander
# Converts a file of ratings into a pickled sparse matrix

import pickle
from scipy.sparse import save_npz, coo_matrix
import pandas as pd


def main():
    print("Converting...")
    in_file = ""
    out_file = ""
    # Load the file
    df = pd.read_csv(in_file)
    # Select the columns
    user_col = df[0]
    item_col = df[1]
    rating_col = df[2]
    # Get the columns as numpy arrays
    users = user_col.to_numpy()
    items = item_col.to_numpy()
    ratings = rating_col.to_numpy()
    # Get the dimensions of the sparse matrix
    num_users = user_col.max()
    num_items = item_col.max()
    # Create the sparse matrix
    sparse_matrix = coo_matrix(((users, items), ratings), [(num_users, num_items)])
    # Save the sparse matrix
    save_npz(out_file, sparse_matrix)


if __name__ == '__main__':
    main()
