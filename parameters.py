class Parameters:
    """
    A data sack for storing parameters in a collective representation.

    """

    def __init__(self, sync: bool, n: int, d: int, k: int, alpha: float, beta: float, lamda: float, normalizer: float, file: str):
        """
        Initializes an instance of Parameters

        :param sync: Sync/Async
        :param n: Number of workers
        :param d: Duration - Iterations for Sync / Seconds for Async
        :param k: Latent factor
        :param alpha: Learning rate
        :param beta: Decay rate
        :param lamda: Normalization factor
        :param file: Path to sparse matrix ratings file
        """
        self.sync = sync
        self.n = n
        self.d = d
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.lamda = lamda
        self.normalizer = normalizer
        self.file = file