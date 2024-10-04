import logging
import sys

# from naps import utils
# import naps.utils

# Basic Information
__name__ = "naps-track"
__version__ = "1.0.9"
__summary__ = "NAPS (NAPS is ArUco Plus SLEAP)"
__url__ = "https://github.com/kocherlab/naps"
__code__ = "https://github.com/kocherlab/naps"
__issue__ = "https://github.com/kocherlab/naps/issues"
__docs__ = "https://naps.readthedocs.io/en/latest/"
__license__ = "MIT"
__copyright__ = "2022"

# Author Information
__authors__ = "Scott Wolf, Dee Ruttenberg, Daniel Knapp, Andrew Webb, Ian Traniello, Grace McKenzie-Smith, Joshua Shaevitz, and Sarah Kocher"
__email__ = "22243650+Wolfffff@users.noreply.github.com, 19213578+aewebb80@users.noreply.github.com"

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
