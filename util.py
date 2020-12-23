from scipy.sparse import load_npz, csr_matrix

DSGD = "DSGD"
DSGDPP = "DSGDPP"
FPSGD = "FPSGD"
NOMAD = "NOMAD"


def load(filename: str, normalize=False) -> csr_matrix:
    print("Loading " + filename)
    try:
        a_csr: csr_matrix = load_npz(filename)
        # Normalize per: https://stackoverflow.com/a/62690439
        normalizer = a_csr.max()
        if normalize:
            a_csr /= normalizer
        print("Loaded {0}".format(filename))
    except IOError:
        print("Could not find file!")
        raise Exception("oops")
    return a_csr


def load_with_features(filename: str, normalize=False) -> (csr_matrix, int, int, float):
    a_csr = load(filename, normalize=normalize)
    shape = a_csr.shape
    normalizer = a_csr.max()
    return a_csr, shape[0], shape[1], normalizer
