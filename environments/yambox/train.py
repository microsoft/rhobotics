#!/usr/bin/env python3
"""Train Rho policies on the Yambox BusyBox dataset.

Yambox is an offline (``environment: null``) training recipe, so no simulation
environment needs to be registered here -- this just forwards to the shared
Rho training entry point.
"""

from rho.train import main

if __name__ == "__main__":
    main()
