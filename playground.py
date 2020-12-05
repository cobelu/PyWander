from scipy.sparse import load_npz


def main():
    in_file = "/Users/cobelu/Desktop/train.npz"
    a_csc = load_npz(in_file)
    print(a_csc)


if __name__ == '__main__':
    main()
