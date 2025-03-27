# -*- coding: utf-8 -*-

from __future__ import annotations
import sys
#from unittest.mock import right
from phenigraph.objet import *
# from phenigraph.gui import plot

if __name__ == '__main__':
    args = sys.argv
    if len(args) > 0:
        tab = np.load(args[1])
        if "list_Graphs" in tab.keys():
            mg = Multigraph(filename=args[1])
            # plot(mg)
            mg.show()
        else:
            gr = Graphique(args[1])
            gr.show()
            # plot(gr)

