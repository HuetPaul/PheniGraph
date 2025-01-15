import matplotlib
from matplotlib.backends.backend_qt import NavigationToolbar2QT

matplotlib.use('Qt5Agg')
import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from phenigraph.objet import *


class PersoNavBar(NavigationToolbar2QT):
    def __init__(self, *args, **kwargs):
        super(PersoNavBar, self).__init__(*args, **kwargs)

    def zoom(self, *args):
        print("args zoom", args)
        super().zoom(*args)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, graph: Graphique | Multigraph):
        if isinstance(graph, Graphique):
            self.axes = graph.axes
            # print(self.axes)
            super().__init__(graph.fig)
        else:
            self.axes = [g.axes for g in graph.list_Graphs]
            # print(self.axes)
            super().__init__(graph.fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, graph: Graphique | Multigraph, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep + 'icone_phenigraph.png'))
        graph.plot()
        sc = MplCanvas(graph)
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        # toolbar = NavigationToolbar(sc, self)
        toolbar = PersoNavBar(sc, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.show()


def plot(graph: Graphique | Multigraph):
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(graph)
    app.exec_()


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
