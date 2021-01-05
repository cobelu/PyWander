import pandas as pd


def main():
    in_file = "/Users/cobelu/Documents/Research/mf/data/ml-25m/train.txt"
    out_file = "~/Desktop/train.csv"
    df = pd.read_csv(in_file, header=None, sep="\t")
    print("Before:\n", df.head())
    # https://stackoverflow.com/a/25652061
    col_list = list(df)
    # TODO: Swap?
    # col_list[0], col_list[1] = col_list[1], col_list[0]
    df = df[col_list]
    print("After:\n", df.head())
    df.sort_values([1, 0], inplace=True)
    print("Sorted:\n", df.head())
    df.to_csv(out_file, header=False, index=False)


if __name__ == '__main__':
    main()
