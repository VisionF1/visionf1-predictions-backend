import sys

# Compatibility shim: alias 'app' to 'visionf1' so that old pickles referencing 'app.core...' can load
sys.modules["app"] = sys.modules[__name__]
