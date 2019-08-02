# -----------------------------------------------------------------------------
# Import
# -----------------------------------------------------------------------------
import argparse
from time import time

from scipy.io import loadmat

from match import matching

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str,
                    help="Path to the file that you want to verify.")
parser.add_argument("--temp_dir", type=str, default="./feature/",
                    help="Path to the directory containing templates.")
parser.add_argument("--thres", type=float, default=0.38,
                    help="Threshold for matching.")
args = parser.parse_args()

if __name__ == '__main__':
    # Extract feature
    start = time()
    print('>>> Start verifying {}\n'.format(args.file))

    # Matching
    ft_load = loadmat(args.file)
    template_extr, mask_extr = ft_load['template'], ft_load['mask']
    result = matching(template_extr, mask_extr, args.temp_dir, args.thres)

    if result == -1:
        print('>>> No registered sample.')

    elif result == 0:
        print('>>> No sample matched.')

    else:
        print('>>> {} samples matched (descending reliability):'.format(len(result)))
        for res in result:
            print("\t", res)

    # Time measure
    end = time()
    print('\n>>> Verification time: {} [s]\n'.format(end - start))
