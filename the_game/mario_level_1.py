#!/usr/bin/env python
__author__ = 'justinarmstrong; lukaszksiezak deep learning mod'

"""
An attempt of implementing deep learning algorithm to win level 1 of
Super Mario.
"""

import sys
import pygame as pg
from data.main import main
import cProfile
import os

if __name__=='__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '-deep':
            main(True)
    else: #run normally 
        main()
        pg.quit()
        sys.exit()