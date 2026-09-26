import argparse
from gmm import CustomGMM

parser = argparse.ArgumentParser(description="Parse GMM parameters")
parser.add_argument("--data", required=True)
parser.add_argument("--n_components", type=int, required=True, help="Set number of Gaussian components")
parser.add_argument("--max_iter", type=int, defalt=100)
parser.add_argument("--random_state", type=int, default=42)
parser.add_argument("--verbose", default=True)
args = parser.parse_args()

X = args.data
gmm = CustomGMM(
    n_components=args.n_components,
    max_iter=args.max_iter,
    random_state=args.random_state,
    verbose=args.verbose)

gmm.fit(X)