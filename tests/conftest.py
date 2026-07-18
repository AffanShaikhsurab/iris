from pathlib import Path

from hypothesis import settings
from hypothesis.database import DirectoryBasedExampleDatabase

settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    database=DirectoryBasedExampleDatabase(Path(".hypothesis/examples")),
    print_blob=True,
)
