from phenigraph import *
if __name__ == '__main__':
    args = sys.argv
    if len(args) > 0:
        tab = np.load(args[1])
        if "list_Graphs" in tab.keys():
            mg = Multigraph(filename=args[1])
            plot(mg)
        else:
            gr = Graphique(args[1])
            plot(gr)