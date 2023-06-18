# Description: This file contains functions for the project.
import os


def _create_directory(dirname):
    if os.path.exists("../bld") == True:
        if os.path.exists(f"../bld/{dirname}") == True:
            pass
        else:
            os.makedirs(f"../bld/{dirname}")
    else:
        os.makedirs("../bld")
        os.makedirs(f"../bld/{dirname}")
