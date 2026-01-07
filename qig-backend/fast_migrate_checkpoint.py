"""
DEPRECATED: Use fast_migrate_vocab_checkpoint.py instead.
"""
import warnings
warnings.warn(
    "fast_migrate_checkpoint.py is deprecated. "
    "Use fast_migrate_vocab_checkpoint.py instead.",
    DeprecationWarning,
    stacklevel=2
)
from fast_migrate_vocab_checkpoint import *

if __name__ == "__main__":
    from fast_migrate_vocab_checkpoint import main
    main()
