"""Settings every test module shares, made before any of them imports a dependency."""

import os

# The tests make no network calls, and Haystack sends usage telemetry to its own servers unless
# this is set when it is imported. A value already in the environment is left as it is.
os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")
