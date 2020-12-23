class Parameters:
    """
    A data sack for storing parameters in a collective representation.

    """

    def __init__(self, sync: bool, n: int, d: int, k: int, alpha: float,
                 beta: float, lamda: float, ptns: int, report: int, filename: str,
                 normalize: bool, bold: bool, verbose: bool, method: str):
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
        self.ptns = ptns
        self.report = report
        self.filename = filename
        self.normalize = normalize
        self.bold = bold
        self.method = method
        self.verbose = verbose
