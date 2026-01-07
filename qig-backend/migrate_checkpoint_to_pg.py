"""
DEPRECATED: Use migrate_vocab_checkpoint_to_pg.py instead.
"""
import warnings
warnings.warn(
    "migrate_checkpoint_to_pg.py is deprecated. "
    "Use migrate_vocab_checkpoint_to_pg.py instead.",
    DeprecationWarning,
    stacklevel=2
)
from migrate_vocab_checkpoint_to_pg import *

if __name__ == "__main__":
    from migrate_vocab_checkpoint_to_pg import main
    main()
