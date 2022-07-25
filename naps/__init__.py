# Basic Information
__title__ = "naps"
__version__ = "0.1"
__summary__ = "NAPS (NAPS is ArUco Plus SLEAP)"
__url__ = "https://github.com/kocherlab/naps"
__code__ = "https://github.com/kocherlab/naps"
__issue__ = "https://github.com/kocherlab/naps/issues"
__docs__ = "https://github.com/kocherlab/naps"
__license__ = "MIT"
__copyright__ = "2022"

# Author Information
__author__ = "Scott Wolf, Dee Ruttenberg, Daniel Knapp, Andrew Webb, and Sarah Kocher"
__email__ = "scott.w.wolf1@gmail.com, 19213578+aewebb80@users.noreply.github.com"

# Maintainer Information
__maintainer__ = ("Scott Wolf, Andrew Webb",)
__maintainer_email__ = (
    "scott.w.wolf1@gmail.com, 19213578+aewebb80@users.noreply.github.com",
)

# Start
import sys
import logging

logging.basicConfig(stream=sys.stdout,
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
