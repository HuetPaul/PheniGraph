from distutils.command.config import config

import numpy as np
from matplotlib import axes
from matplotlib.collections import LineCollection
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import LogNorm
from matplotlib.colors import to_rgba
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as sp
from matplotlib.pyplot import colorbar
from scipy.interpolate import griddata
from scipy.stats import alpha

ii = np.iinfo(int)  # Information on numpy's integer
ii_max = ii.max  # The maximum int numerically possible

c: str = "ysqgsdkbn"  # Chain used to fill the gaps in the lists during the saving of the Graphique

#: Defaults colors :
C1: str = '#6307ba'  #: Violet / Purple
C2: str = '#16b5fa'  #: Cyan
C3: str = '#2ad500'  #: Vert clair / Light green
C4: str = '#145507'  #: vert foncé / Dark green
C5: str = '#ff8e00'  #: Orange
C6: str = '#cb0d17'  #: Rouge / Red
C7: str = '#5694b2'  #: Bleu pastel / Pastel blue
C8: str = '#569a57'  #: Vert pastel / Pastel green
C9: str = '#b986b9'  #: Lavande
C10: str = '#c6403c'  #: Rouge pastel / Pastel red
C11: str = '#d39d5d'  #: Beige
C12: str = '#25355d'  #: Bleu / Blue
C13: str = '#fcc100'  #: Jaune / Yellow
C14: str = '#7ab5fa'  #: Bleu ciel / Light blue
C15: str = '#fc2700'  #: Orange foncé / Dark orange
C16: str = '#0fc88f'  #: Bleu-Vert / Blue-Green
C17: str = '#a8173b'  #: Rouge cerise / Red
C18: str = '#1812c4'  #: Bleu foncé / Dark blue
C19: str = "#000000"  #: Noir / Black
C20: str = "#707070"  #: Gris / Grey
Ctrensp: tuple = (0, 0, 0, 0)  #: completely trensparent color (to use for hiding curves for exemple)
l_colors: list[str] = [C1, C2, C4, C3, C5, C6, C7, C8, C9,
                       C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20]


def linear_color_interpolation(val: np.float64 | float | list[np.float64 | float] | np.ndarray[np.float64 | float],
                               val_min: np.float64 = - np.inf, val_max: np.float64 = np.inf,
                               col_min: str | tuple = C1, col_max: str = C2,
                               ) -> str | list[str] | np.ndarray[str]:
    """

    Return a color/list of colors which is linearly interpolated between the two extremal colors col_min
    and col_max

    Parameters
    ----------
    val: np.float64 | float | list[np.float64 | float] | np.ndarray[np.float64 | float],
        the value to be interpolated
    val_min: np.float64, default = min(val)
        the minimal value
    val_max: np.float64, default = max(val)
        the maximal value
    col_min: str | tuple, optional, default=C1=#6307ba (purple)
        color associated with the minimal value (in hexadecimal or rgba)
    col_max: str | tuple, optional, default=C2=#16b5fa (cyan)
        color associated with the maximal value (in hexadecimal or rgba)

    Returns
    -------
    str | list[str] | np.ndarray[str]
        the interpolated color(s) in hex

    """
    if isinstance(col_min, str):
        col_min: tuple = to_rgba(col_min)
    if isinstance(col_max, str):
        col_max: tuple = to_rgba(col_max)
    col_min: np.ndarray = np.array(col_min)
    col_max: np.ndarray = np.array(col_max)

    if val_min == - np.inf:
        val_min = np.min(val)
    if val_max == np.inf:
        val_max = np.max(val)

    if (isinstance(val, np.float64) or isinstance(val, float)
            or isinstance(val, np.int64) or isinstance(val, int)):
        return to_hex(tuple(col_min + (col_max - col_min) * (val - val_min) / (val_max - val_min)))
    else:
        res: list[str] = []
        for v in val:
            res.append(to_hex(tuple(np.minimum(np.double(1.),
                                               np.maximum(np.double(0.), col_min + (col_max - col_min)
                                                          * (v - val_min) / (val_max - val_min))))))

    if isinstance(val, np.ndarray):
        return np.array(res)
    elif isinstance(val, list):
        return res
    else:
        raise UserWarning("The values to be interpolated has the wrong type :", type(val),
                          "the only accepted types are float, np.float64, list[np.float64| float], np.ndarray")


def test_list_regular(x: list) -> list | None:
    """
    If x is a list of lists, this function test if there size is equals.
    If there are, then it returns a list of string of the same dimensions of x.
    Then x could be turned into a ndarray and turned back after

    Parameters
    ----------
    x: list
        the list to be tested

    Returns
    -------
        the type list or None
    """
    res: list = []
    dim: int = -1  # size of the sub list (-1 for scalars or strings)
    if len(x) == 0:
        return []
    elif isinstance(x[0], list | np.ndarray):
        dim = len(x[0])
    for X in x:
        if isinstance(X, str | float | int) and dim > -1:
            return None
        elif isinstance(X, list | np.ndarray) and len(X) != dim:
            return None
        if isinstance(X, str):
            res.append("str")
        elif isinstance(X, np.float64):
            res.append("double")
        elif isinstance(X, float):
            res.append("float")
        elif isinstance(X, np.int64):
            res.append("int64")
        elif isinstance(X, int):
            res.append("int")
        elif isinstance(X, tuple):
            res.append("tuple")
        elif isinstance(X, np.ndarray):
            res.append("array")
        elif isinstance(X, list):
            types: list | None = test_list_regular(X)
            if types is None:
                return None
            else:
                res.append(types)
        elif isinstance(X, dict):
            return None
        elif isinstance(X, Graphique):
            return None
        else:
            raise UserWarning("test_list_regular : the type ", type(X), "cannot be saved")
    return res


def get_regular_list(x: np.ndarray | list, types: np.ndarray) -> list:
    """
    Recover the original list converted by test_list_regular (if the list is regular (the result is not None))

    Parameters
    ----------
    x: np.ndarray | list
        The converted array
    types: np.ndarray
        The types of each list's element (result of 'test_list_regular')

    Returns
    -------
    dict
        the original list

    """
    res: list = []
    for (X, T) in zip(x, types):
        if isinstance(X, list) or isinstance(X, np.ndarray) and not isinstance(T, str):
            res.append(get_regular_list(X, T))
        elif isinstance(X, np.ndarray) and len(X) == 0:
            res.append([])
        elif T == "str":
            res.append(str(X))
        elif T == "double":
            res.append(np.double(X))
        elif T == "float":
            res.append(float(X))
        elif T == "int":
            res.append(int(X))
        elif T == "int64":
            res.append(np.int64(X))
        elif T == "tuple":
            res.append(tuple(X))
        elif T == "array":
            res.append(X)
        else:
            raise UserWarning("get_regular_list : the type ", T, "cannot be loaded")
    return res


def list_to_dict(x: list, separator: str = "-.-") -> dict:
    """
    Parameters
    ----------
    x: list
        the list to be converted
    separator: str, optinal, default="-.-"
        The string to be added between dictionary's keys if there are recursives

    Returns
    -------
    dict
        the dictionary containing the list

    """
    res: dict = dict()
    types: list[list[str]] = []
    for i in range(len(x)):
        if isinstance(x[i], str):
            res[str(i)] = x[i]
            types.append([str(i), "str"])
        elif isinstance(x[i], np.float64):
            res[str(i)] = x[i]
            types.append([str(i), "double"])
        elif isinstance(x[i], float):
            res[str(i)] = x[i]
            types.append([str(i), "float"])
        elif isinstance(x[i], np.int64):
            res[str(i)] = x[i]
            types.append([str(i), "int64"])
        elif isinstance(x[i], int):
            res[str(i)] = x[i]
            types.append([str(i), "int"])
        elif isinstance(x[i], tuple):
            res[str(i)] = x[i]
            types.append([str(i), tuple])
        elif isinstance(x[i], np.ndarray):
            res[str(i)] = x[i]
            types.append([str(i), "array"])
        elif isinstance(x[i], list):
            loc_types: list | None = test_list_regular(x[i])
            if loc_types is not None:
                res[str(i)] = np.array(x[i])
                types.append([str(i), "list_regular"])
                res[str(i) + separator + "types"] = np.array(loc_types)
                types.append([str(i) + separator + "types", "types"])
            else:
                loc_dic: dict = list_to_dict(x[i], separator=separator)
                types.append([str(i), "list_irregular"])
                res[str(i)] = str(i)
                for lk in loc_dic.keys():
                    res[str(i) + separator + lk] = loc_dic[lk]
                    types.append([str(i) + separator + lk, "list_irregular_values"])
        elif isinstance(x[i], dict):
            loc_dic: dict = dict_to_ndarray_dict(x[i], separator=separator)
            types.append([str(i), "dict"])
            res[str(i)] = str(i)
            for lk in loc_dic.keys():
                res[str(i) + separator + lk] = loc_dic[lk]
                types.append([str(i) + separator + lk, "dict_values"])
        elif isinstance(x[i], Graphique):
            loc_dic: dict = dict_to_ndarray_dict(x[i].to_dict(), separator=separator)
            types.append([str(i), "Graphique"])
            res[str(i)] = str(i)
            for lk in loc_dic.keys():
                res[str(i) + separator + lk] = loc_dic[lk]
                types.append([str(i) + separator + lk, "Graphique_values"])
        else:
            raise UserWarning("list_to_dict : the type ", type(x[i]), "cannot be saved")
    res["types"] = types
    res["separator"] = separator

    return res


def dict_to_ndarray_dict(dic: dict, separator: str = "-.-") -> dict:
    """
    Turn a dictionary containing list, ndarray, str, float, int or even others dic into a dic in the
    right form to be saved by np.save_compressed

    Parameters
    ----------
    dic: dict
        The dictionary to be transformed
    separator: str, optional, default="-.-"
        The string to be added between dictionary's keys if there are recursives

    Returns
    -------
    dict
        The new dictionary

    """
    res: dict = dict()
    types: list[[str, str]] = []
    keys: list[str] = []

    for k in dic.keys():
        if "k" == "types":
            raise UserWarning("""dict_to_ndarray_dict : the key "types" cannot be saved, 
            please replace it with another one""")
        if isinstance(dic[k], str):
            res[k] = dic[k]
            types.append([k, "str"])
            keys.append(k)
        elif isinstance(dic[k], np.int64):
            res[k] = dic[k]
            types.append([k, "int64"])
            keys.append(k)
        elif isinstance(dic[k], int):
            res[k] = dic[k]
            types.append([k, "int"])
            keys.append(k)
        elif isinstance(dic[k], np.float64):
            res[k] = dic[k]
            types.append([k, "double"])
            keys.append(k)
        elif isinstance(dic[k], float):
            res[k] = dic[k]
            types.append([k, "float"])
            keys.append(k)
        elif isinstance(dic[k], tuple):
            res[k] = dic[k]
            types.append([k, "tuple"])
            keys.append(k)
        elif isinstance(dic[k], np.ndarray):
            res[k] = dic[k]
            types.append([k, "array"])
            keys.append(k)
        elif isinstance(dic[k], list):
            loc_types: list | None = test_list_regular(dic[k])
            if loc_types is not None:
                res[k] = np.array(dic[k])
                types.append([k, "list_regular"])
                res[k + separator + "types"] = np.array(loc_types)
                types.append([k + separator + "types", "types"])
            else:
                loc_dic: dict = list_to_dict(dic[k], separator=separator)
                types.append([k, "list_irregular"])
                res[k] = k
                for lk in loc_dic.keys():
                    res[k + separator + lk] = loc_dic[lk]
                    types.append([k + separator + lk, "list_irregular_values"])
        elif isinstance(dic[k], dict):
            loc_dic: dict = dict_to_ndarray_dict(dic[k], separator=separator)
            types.append([k, "dict"])
            res[k] = k
            for lk in loc_dic.keys():
                res[k + separator + lk] = loc_dic[lk]
                types.append([k + separator + lk, "dict_values"])
        elif isinstance(dic[k], Graphique):
            loc_dic: dict = dict_to_ndarray_dict(dic[k].to_dict(), separator=separator)
            types.append([k, "Graphique"])
            res[k] = k
            for lk in loc_dic.keys():
                res[k + separator + lk] = loc_dic[lk]
                types.append([k + separator + lk, "Graphique_values"])
        else:
            raise UserWarning("dict_to_ndarray_dic : the type ", type(dic[k]), "cannot be saved")
    res["types"] = types
    res["separator"] = separator
    return res


def dict_to_list(dic: dict) -> list:
    """
    Return the irregular list (A list with list of differents sizes or with dictionary(s))
    transformed by the list_to_dict function

    Parameters
    ----------
    dic: dict
        The dic to be converted

    Returns
    -------
    list
        The original list

    """
    if "separator" not in dic.keys():
        raise UserWarning("dic_to_list, the given dictionary doesn't contain the required separator key."
                          "Either this dict is not the result of list_to_dict either it has been modified")

    separator: str = str(dic["separator"])
    del dic["separator"]
    types = dic["types"]
    del dic["types"]
    num_keys: list = []
    num_keys_int: list = []
    keys: list[str] = list(dic.keys())
    keys.sort()
    for k in keys:
        if separator in k:
            num_keys_int.append(int(k[:k.find(separator)]))
            num_keys.append(np.double(k[:k.find(separator)])
                            + np.double(num_keys_int.count(int(k[:k.find(separator)])) - 1) * 1e-6)
        else:
            num_keys_int.append(int(k))
            num_keys.append(np.double(k)
                            + np.double(num_keys_int.count(int(k)) - 1) * 1e-6)
    keys: np.ndarray[str] = np.array(list(dic.keys()))
    keys = keys[np.argsort(num_keys)]
    res: list = []
    i: int = 0
    # while i < len(keys):
    #     k: str = keys[i]
    #     if k == "types" or k == "separator":
    #         i += 1
    #     else:  # this is an original element of the list
    while i < len(keys):
        # k = str(ky)
        # print("key : ", k)
        k: str = keys[i]
        type_i = types[np.argwhere(types[:, 0] == k)[0, 0]][1]
        if type_i == "str":
            res.append(str(dic[k]))
            i += 1
        elif type_i == "double":
            res.append(np.double(dic[k]))
            i += 1
        elif type_i == "float":
            res.append(float(dic[k]))
            i += 1
        elif type_i == "int64":
            res.append(np.int64(dic[k]))
            i += 1
        elif type_i == "int":
            res.append(int(dic[k]))
            i += 1
        elif type_i == "tuple":
            res.append(tuple(dic[k]))
            i += 1
        elif type_i == "array":
            res.append(np.array(dic[k]))
            i += 1
        elif type_i == "list_regular":
            res.append(get_regular_list(dic[k], dic[k + separator + "types"]))
            i += 2
        elif type_i == "list_irregular":
            loc_dic: dict = dict()
            i += 1
            while i < len(keys) and k + separator in keys[i]:
                loc_dic[keys[i][len(k) + len(separator):]] = dic[keys[i]]
                i += 1
            res.append(dict_to_list(loc_dic))
        elif type_i == "dict":
            loc_dic: dict = dict()
            i += 1
            while i < len(keys) and k + separator in keys[i]:
                loc_dic[keys[i][len(k) + len(separator):]] = dic[keys[i]]
                i += 1
            res.append(ndarray_dict_to_dict(loc_dic))
        elif type_i == "Graphique":
            loc_graph: Graphique = Graphique()
            loc_dic: dict = dict()
            i += 1
            while i < len(keys) and k + separator in keys[i]:
                loc_dic[keys[i][len(k) + len(separator):]] = dic[keys[i]]
                i += 1
            loc_graph.load_dict(ndarray_dict_to_dict(loc_dic))
            res.append(loc_graph)
        else:
            raise UserWarning("dict_list : the type ", type_i, "cannot be loaded", k)
    return res


def ndarray_dict_to_dict(dic: dict) -> dict:
    """
    Return the original dictionary get from the dict_to_ndarray_dict

    Parameters
    ----------
    dic, dict
        the dictionary to be converted

    Returns
    -------
    dict
        the original dictionary

    """
    if "separator" not in dic.keys():
        raise UserWarning("dic_to_list, the given dictionary doesn't contain the required separator key."
                          "Either this dict is not the result of list_to_dict either it has been modified")

    separator: str = str(dic["separator"])
    if "types" not in dic.keys():
        raise UserWarning("dic_to_list, the given dictionary doesn't contain the required _types key."
                          "Either this dict is not the result of list_to_dict either it has been modified",
                          dic.keys())
    types: np.ndarray = dic["types"]
    keys: list[str] = list(dic.keys())
    keys.sort()
    res: dict = dict()
    i: int = 0
    while i < len(keys):
        k: str = keys[i]
        if k == "types" or k == "separator":
            i += 1
        else:  # this is an original element of the list
            type_i = types[np.argwhere(types[:, 0] == k)[0, 0]][1]
            if type_i == "str":
                res[k] = str(dic[k])
                i += 1
            elif type_i == "double":
                res[k] = np.double(dic[k])
                i += 1
            elif type_i == "float":
                res[k] = float(dic[k])
                i += 1
            elif type_i == "int64":
                res[k] = np.int64(dic[k])
                i += 1
            elif type_i == "int":
                res[k] = int(dic[k])
                i += 1
            elif type_i == "tuple":
                res[k] = tuple(dic[k])
                i += 1
            elif type_i == "array":
                res[k] = np.array(dic[k])
                i += 1
            elif type_i == "list_regular":
                res[k] = get_regular_list(dic[k], dic[k + separator + "types"])
                i += 2
            elif type_i == "list_irregular":
                loc_dic: dict = dict()
                i += 1
                while k + separator in keys[i]:
                    loc_dic[keys[i][len(k) + len(separator):]] = dic[keys[i]]
                    i += 1
                res[k] = dict_to_list(loc_dic)
            elif type_i == "dict":
                loc_dic: dict = dict()
                i += 1
                while k + separator in keys[i]:
                    loc_dic[keys[i][len(k) + len(separator):]] = dic[keys[i]]
                    i += 1
                res[k] = ndarray_dict_to_dict(loc_dic)
            elif type_i == "Graphique":
                loc_graph: Graphique = Graphique()
                loc_dic: dict = dict()
                i += 1
                while k + separator in keys[i]:
                    loc_dic[keys[i][len(k) + len(separator):]] = dic[keys[i]]
                    i += 1
                loc_graph.load_dict(ndarray_dict_to_dict(loc_dic))
                res[k] = loc_graph
            else:
                raise UserWarning("dict_list : the type ", type_i, "cannot be loaded")
    return res


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    This function is adapted from https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    # if "array" in lc_kwargs:
    #     warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)
    lc = LineCollection(segments, color=c, **default_kwargs)
    # lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


# noinspection PyTypeChecker
class Graphique:
    """
The purpose of this object is to make it easier to create and manipulate graphs
It contains all the variables required to plot a graph on an axis
It can be used to display the graph, save it as a .png file and save it using a
.npy format in order to avoid having to easily regenerate the graph in a reproducible way.

------------------------Handling procedures----------------------------

To initialise a Graphique :
    - For a new Graphique: ```gr=Graphique()```
    - To open a Graphique with name ```n``` in directory ```f``` :
        (```n``` and ```f``` are two strings, by default ```f``` is the current working directory)
        e.g. ```f="../Images/Graphics"``` and ```n="test_graphique"```.
        (the ```.npz``` extension is not mandatory) :
        - ```gr=Graphique(n)``` if ```n``` is in the current directory
        - ```gr=Graphique(n, f)``` otherwise

To save a Graphique :
    - Assign a name to the graphic, without using an extension :
        ```g.filename = "new_name"```
        The default name is graph_without_name. If you want to save several Graphiques in the same folder
        it is therefore important to give them a name (not automatically)
    - If necessary, assign a directory for saving:
        ```g.directory=‘new_directory’``` By default, the location is the current working directory.
    - To save the object :
        ```g.save()```
    - To save the figure :
        - Assign an extension if necessary (by default the extension is svg).
            Possible extensions are those available via the matplotlib library: ‘png’, ‘pdf’, ‘svg’, etc.
            ```g.ext=".new_extension"```
        - ```g.save_figure()```

To show the Graphique :
    - ```g.show()```

To add a line (equivalent to plt.plot) :
    - ```g.add_line(x, y, **args)```

    with x and y the list(s)/ndarray to plot
    and ```**args``` are all the ather arguments of plt.plot()
    Can be repeated as many times as required

To add a histogram :
    - ```g.add_histogram(values, weight=[], normalisation=True, statistic=‘sum’, bins=10, range=None, **args)``` :

    where
        - values is the array, the list of values to classify
        - weights is a list giving a possible weight for each value
        - normalisation indicates whether the histogram should be normalised
        - The other arguments are the same as ```plt.hist()```

    Can be repeated as many times as necessary

To add an image :
    - ```g.add_image(array,x_axis,y_axis,**args)```

    where :
        - ```array``` represents the image to be displayed
        - ```axis_x``` and ```axis_y``` give the graduations of the image axes
        - ```**args``` all the other possible arguments for displaying an image

To add contours :
    - ```g.add_contours(self, contours=np.array([[]]), x_axe=None, y_axe=None, **args)``` :

   where
        - ```**args``` gives the possible arguments for ```plt.contours()```
        - To add level lines to an image complete ```**args```, leaving the other arguments by default

To add a polygon (coloured area delimited by a list of points) :
    - ```g.add_polygon(ind,alpha0=0.7, facecolor=‘C3’,**args)```

with ```ind``` an array/list of dimension (n,2) where n is the number of points.
```ind[:,0]``` corresponds to the abscissas of the points and
```ind[:1]``` corresponds to the to the ordinates.
Can be repeated as many times as necessary

To display several Graphique in one, use a Multigraph

    """

    def __init__(self, filename: str = "", directory: str = ""):
        if '.npz' in filename:
            filename = filename[:-4]
        self.filename: str = filename

        if filename == "":
            self.directory: str = "./"
            self.filename: str = "graph_without_name"
        self.ext: str = ".png"
        self.title: str = ""
        self.style: str = 'default'  # global style:
        # styles available: plt.style.available : 'default' 'Solarize_Light2'
        # '_classic_test_patch' '_mpl-gallery' '_mpl-gallery-nogrid'
        # 'bmh' 'classic' 'dark_background' 'fast' 'fivethirtyeight' 'ggplot'
        # 'grayscale' 'seaborn' 'seaborn-bright' 'seaborn-colorblind' 'seaborn-dark' 'seaborn-dark-palette'
        # 'seaborn-darkgrid' 'seaborn-deep' 'seaborn-muted' 'seaborn-notebook' 'seaborn-paper' 'seaborn-pastel'
        # 'seaborn-poster' 'seaborn-talk' 'seaborn-ticks' 'seaborn-white' 'seaborn-whitegrid' 'tableau-colorblind10'
        self.param_font: dict = {
            "font.size": 13}  # Contains additional parameters for global font management
        self.ax: axes = None  # x axis on the bottom, y axis on the left
        self.ax_tl: axes = None  # x axis on the top, y axis on the left
        self.ax_br: axes = None  # x axis on the bottom, y axis on the right
        self.ax_tr: axes = None  # x axis on the top, y axis on the right
        self.axes: list[axes] = None
        self.param_ax: dict = dict()
        self.param_ax_tl: dict = dict()
        self.param_ax_br: dict = dict()
        self.param_ax_tr: dict = dict()
        self.colorbar: list[Colorbar] = []
        self.cmap: list = []
        self.norms: list = []
        self.param_colorbar: list[dict] = []
        self.ticks_colorbar: list[np.ndarray | list] = []
        # Contains additional parameters for the colorbars ex : label="legende"...
        # The first one is automatically associated with the image
        self.custum_colorbar_colors: list[list] = None
        # To bild custums discrete colorbars defind by the colors in thoses list (one list for one colorbar)
        self.custum_colorbar_values: list[list] = None
        # The values associated with the colors of self.custum_colorbar_colors
        self.index_colorbar_image: list[int] = []  # Index of the image's colorbar(s)
        self.fig: Figure = None
        self.param_fig: dict = dict()  # Contains additional parameters for the Figure ex : facecolor="w"...
        self.param_enrg_fig: dict = dict(bbox_inches="tight")
        # Contains additional parameters to save the Figure ex : dpi=300...

        # orders of axis in the lists : ["bl", "tl, "tr", "br"]
        # liste of position of x-axis ticks
        self.x_axe: np.ndarray[np.ndarray[float | np.float64]] = np.array([np.array([-np.inf]), np.array([-np.inf]),
                                                                           np.array([-np.inf]), np.array([-np.inf])])
        # List of x-ticks labels if not "empty"
        self.labels_x_ticks: np.ndarray[np.ndarray[str]] = np.array([np.array(["empty"]), np.array(["empty"]),
                                                                     np.array(["empty"]), np.array(["empty"])])

        # liste of position of y-axis ticks
        self.y_axe: np.ndarray[np.ndarray[float | np.float64]] = np.array([np.array([-np.inf]), np.array([-np.inf]),
                                                                           np.array([-np.inf]), np.array([-np.inf])])
        # List of y-ticks labels if not "empty"
        self.labels_y_ticks: np.ndarray[np.ndarray[str]] = np.array([np.array(["empty"]), np.array(["empty"]),
                                                                     np.array(["empty"]), np.array(["empty"])])

        self.lines_x: list[
            list[float | np.float64]] = []  # list containing lists of x coordinates to be displayed via plt.plot
        self.lines_y: list[
            list[float | np.float64]] = []  # list containing lists of y coordinates to be displayed via plt.plot
        self.lines_t_x: list[
            list[float | np.float64]] = []  # list containing lists of x coordinates to be displayed via plt.text
        self.lines_t_y: list[
            list[float | np.float64]] = []  # list containing lists of y coordinates to be displayed via plt.text
        self.lines_t_s: list[
            list[float | np.float64]] = []  # list containing lists of string to be displayed via plt.text
        self.err_x: list[list[
            float | np.float64]] = []  # list containing lists of errors associated with x coordinates
        self.err_y: list[list[
            float | np.float64]] = []  # list containing lists of errors associated with y coordinates
        self.compt_color: int = -1  # Indicate the index of the last color used in line in the list_colors
        # Additianal parameter's dictionarys
        self.param_lines: list = []  # for lines
        self.param_texts: list = []  # for texts
        # Order of line plot (default the order of saving)
        self.indexs_plot_lines: list = []
        # histogrammes
        self.bords_histogramme: list = []  # List of lists of coordinates of the borders of histograms' bars
        # liste contenant les valeurs des hauteurs de barres des histogrammes
        self.vals_histogramme: list = []  # List of lists of the values of each histograms' bars
        self.param_histogrammes: list = []  # Additional parameters dictionary for plt.bar

        self.param_legende: dict = dict()  # parameters of plt.legende

        # Table to plot via plt.pcolor, the limites are x_axe/y
        self.array_image: np.ndarray = np.array([[]])
        self.x_axe_image: np.ndarray = np.array([])  # list of x-coordinate for the image
        self.y_axe_image: np.ndarray = np.array([])  # list of y-coordinate for the image

        # Parameters of pcolor colorbar,legend...
        self.param_image: dict = dict()  # for pcolor
        self.array_contours: np.ndarray = np.array([[]])
        # table to plot via contour, the limites are x_axe/y switch of half a bin

        self.clabels: np.ndarray = np.array([])  # Label to plot for each contours levels
        self.clabels_mask: np.ndarray = np.array([])  # Mask or index list of labels to plot for contours
        self.param_contours: dict = dict()  # parameter of contours : alpha0,label...
        self.param_labels_contours: dict = dict(fontsize=15, inline=True)
        # Contains additional parameters for the labels of the contours
        self.color_label_contours: list[str] = []
        self.x_axe_contours: np.ndarray = np.array([])  # liste of x-coordinate for contours
        self.y_axe_contours: np.ndarray = np.array([])  # liste of y-coordinate for contours
        # lists of levels for contours to plot
        self.levels: np.ndarray = np.array([])
        # number of levels for contours to plot, ignore if levels != np.array([])
        self.nb_contours: int = 10
        self.tab_contours_is_image: bool = False  # If True, the table used for contour is self.image

        self.index_polygons: list = []  # Index of polygone polygons to plot
        self.param_polygons: list = []  # Parameters for the polygons plotting
        self.grid: bool = False  # If True add a background grid via plt.grid()
        if filename != "":
            self.filename = filename
            if directory != "":
                self.directory = directory
            elif "/" in directory:
                i: int = directory.find("/")
                while directory.find("/") > 0:
                    # Cherche la dernière occurrence de "/"
                    i = directory.find("/")
                self.directory = filename[:i]
                self.filename = filename[i:]
            else:
                self.directory = "./"
            values_files: np.lib.npyio.NpzFile = np.load(directory + filename + ".npz")
            values_to_load: dict = ndarray_dict_to_dict(dict(values_files))
            values_files.close()
            self.load_dict(values_to_load)

    def load_dict(self, values_to_load: dict) -> None:
        """
        Load a Graphique contained in a dictionary

        Parameters
        ----------
        values_to_load: dict
            The dictionary that contain all the necessery informations to build a Graphique.
            Should be produced by another Graphique via Graphique.to_dict()

        Returns
        -------
        None

        See Also
        --------
        Graphique.to_dict()

        """
        if "ext" in values_to_load.keys():
            self.ext = values_to_load["ext"]
        self.ax = None
        self.ax_tl = None
        self.ax_br = None
        self.ax_tr = None
        self.axes = None
        self.param_ax = dict()
        if 'style' in values_to_load.keys():
            self.style = values_to_load["style"]
        if "param_colorbar" in values_to_load.keys():
            self.param_colorbar = values_to_load["param_colorbar"]

        if "ticks_colorbar" in values_to_load.keys():
            self.ticks_colorbar = values_to_load["ticks_colorbar"]
        if "custum_colorbar_colors" in values_to_load.keys():
            self.custum_colorbar_colors = values_to_load["custum_colorbar_colors"]
        if "custum_colorbar_values" in values_to_load.keys():
            self.custum_colorbar_values = values_to_load["custum_colorbar_values"]
        if "index_colorbar_image" in values_to_load.keys():
            self.index_colorbar_image = values_to_load["index_colorbar_image"]
        if "param_labels_contours" in values_to_load.keys():
            self.param_labels_contours = values_to_load["param_labels_contours"]
        if "color_label_contours" in values_to_load.keys():
            self.color_label_contours = values_to_load["color_label_contours"]
        if "param_font" in values_to_load.keys():
            self.param_font = values_to_load["param_font"]
        if "param_ax" in values_to_load.keys():
            self.param_ax = values_to_load["param_ax"]
        if "param_ax_tl" in values_to_load.keys():
            self.param_ax_tl = values_to_load["param_ax_tl"]
        if "param_ax_br" in values_to_load.keys():
            self.param_ax_br = values_to_load["param_ax_br"]
        if "param_ax_tr" in values_to_load.keys():
            self.param_ax_tr = values_to_load["param_ax_tr"]
        # self.colorbar = None
        if "param_fig" in values_to_load.keys():
            self.param_fig = values_to_load["param_fig"]
        if "param_enrg_fig" in values_to_load.keys():
            self.param_enrg_fig = values_to_load["param_enrg_fig"]
        if "x_axe" in values_to_load.keys():
            if len(values_to_load["x_axes"].shape) == 2:
                self.x_axe = values_to_load["x_axe"]
            else:
                self.x_axe = np.array([values_to_load["x_axe"], np.array([-np.inf]),
                                       np.array([-np.inf]), np.array([-np.inf])])
        if "labels_x_ticks" in values_to_load.keys():
            if len(values_to_load["labels_x_ticks"].shape) == 2:
                self.labels_x_ticks = values_to_load["labels_x_ticks"]
            else:
                self.labels_x_ticks = np.array([values_to_load["labels_x_ticks"], np.array(["empty"]),
                                                np.array(["empty"]), np.array(["empty"])])
        if "y_axe" in values_to_load.keys():
            if len(values_to_load["y_axes"].shape) == 2:
                self.y_axe = values_to_load["y_axe"]
            else:
                self.y_axe = np.array([values_to_load["y_axe"], np.array([-np.inf]),
                                       np.array([-np.inf]), np.array([-np.inf])])

        if "labels_y_ticks" in values_to_load.keys():
            if len(values_to_load["labels_y_ticks"].shape) == 2:
                self.labels_y_ticks = values_to_load["labels_y_ticks"]
            else:
                self.labels_y_ticks = np.array([values_to_load["labels_y_ticks"], np.array(["empty"]),
                                                np.array(["empty"]), np.array(["empty"])])

        if "lines_x" in values_to_load.keys():
            self.lines_x = values_to_load["lines_x"]
        if "lines_y" in values_to_load.keys():
            self.lines_y = values_to_load["lines_y"]
        if "indexs_plot_lines" in values_to_load.keys():
            self.indexs_plot_lines = values_to_load["indexs_plot_lines"]
        if "lines_t_x" in values_to_load.keys():
            self.lines_t_x = values_to_load["lines_t_x"]
        if "lines_t_y" in values_to_load.keys():
            self.lines_t_y = values_to_load["lines_t_y"]
        if "lines_t_s" in values_to_load.keys():
            self.lines_t_s = values_to_load["lines_t_s"]
        if "err_x" in values_to_load.keys():
            self.err_x = values_to_load["err_x"]
        if "err_y" in values_to_load.keys():
            self.err_y = values_to_load["err_y"]
        if "compt_color" in values_to_load.keys():
            self.compt_color = values_to_load["compt_color"]
        self.param_lines = []
        if "param_lines" in values_to_load.keys():
            self.param_lines = values_to_load["param_lines"]
        if "param_texts" in values_to_load.keys():
            self.param_texts = values_to_load["param_texts"]
        if "bords_histogramme" in values_to_load.keys():
            self.bords_histogramme = values_to_load["bords_histogramme"]
        if "vals_histogramme" in values_to_load.keys():
            self.vals_histogramme = values_to_load["vals_histogramme"]
        if "param_histogrammes" in values_to_load.keys():
            self.param_histogrammes = values_to_load["param_histogrammes"]
        if "param_legende" in values_to_load.keys():
            self.param_legende = values_to_load["param_legende"]
        if "array_image" in values_to_load.keys():
            self.array_image = values_to_load["array_image"]
            self.param_image = values_to_load["param_image"]
            self.x_axe_image = values_to_load["x_axe_image"]
            self.y_axe_image = values_to_load["y_axe_image"]
        if "contours" in values_to_load.keys():
            self.array_contours = values_to_load["contours"]
            self.x_axe_contours = values_to_load["x_axe_contours"]
            self.y_axe_contours = values_to_load["y_axe_contours"]
        if "param_contours" in values_to_load.keys():
            self.param_contours = values_to_load["param_contours"]
        if "levels" in values_to_load.keys():
            self.levels = values_to_load["levels"]
            if "clabels" in values_to_load.keys():
                self.clabels = values_to_load["clabels"]
            if "clabels_mask" in values_to_load.keys():
                self.clabels_mask = values_to_load["clabels_mask"]
        if "parameters" in values_to_load.keys():
            self.title = values_to_load["parameters"][0]
            self.nb_contours = int(values_to_load["parameters"][1])
            self.tab_contours_is_image = bool(int(values_to_load["parameters"][2]))
            self.grid = bool(int(values_to_load["parameters"][3]))
        if "index_polygons" in values_to_load.keys():
            self.index_polygons = values_to_load["index_polygons"]
            self.param_polygons = values_to_load["param_polygons"]

    def to_dict(self) -> dict:
        """
        Build a dictionary containing all the information of this Graphique
        to save it with np.saved_compressed

        Returns
        -------
        dict

        See Also
        --------
        Graphique.to_dict()
        """
        enrg: dict = dict()  # Dictionary containing all the necessary information :
        # Used like :  np.savez_compressed(name_fichier,**enrg)

        enrg["ext"] = self.ext
        enrg["style"] = self.style
        if len(self.param_colorbar) > 0:
            enrg["param_colorbar"] = self.param_colorbar
        if len(self.ticks_colorbar) > 0:
            enrg["ticks_colorbar"] = self.ticks_colorbar
        if self.custum_colorbar_colors is not None:
            enrg["custum_colorbar_colors"] = self.custum_colorbar_colors
        if self.custum_colorbar_values is not None:
            enrg["custum_colorbar_values"] = self.custum_colorbar_values
        if len(self.index_colorbar_image) > 0:
            enrg["index_colorbar_image"] = self.index_colorbar_image
        if len(self.param_labels_contours) > 0:
            enrg["param_labels_contours"] = self.param_labels_contours
        if len(self.param_font) > 0:
            enrg["param_font"] = self.param_font
        if len(self.param_ax) > 0:
            enrg["param_ax"] = self.param_ax
        if len(self.param_ax_tl) > 0:
            enrg["param_ax_tl"] = self.param_ax_tl
        if len(self.param_ax_br) > 0:
            enrg["param_ax_br"] = self.param_ax_br
        if len(self.param_ax_tr) > 0:
            enrg["param_ax_tr"] = self.param_ax_tr
        if len(self.param_fig) > 0:
            enrg["param_fig"] = self.param_fig
        if len(self.param_enrg_fig) > 0:
            enrg["param_enrg_fig"] = self.param_enrg_fig
        if len(self.x_axe) == 0 or np.any(self.x_axe > -np.inf):
            enrg["x_axe"] = self.x_axe
        if len(self.labels_x_ticks) == 0 or np.any(self.labels_x_ticks != "empty"):
            enrg["labels_x_ticks"] = self.labels_x_ticks
        if len(self.y_axe) == 0 or np.any(self.y_axe > -np.inf):
            enrg["y_axe"] = self.y_axe
        if len(self.labels_y_ticks) == 0 or np.any(self.labels_y_ticks != "empty"):
            enrg["labels_y_ticks"] = self.labels_y_ticks
        if len(self.lines_x) > 0:
            enrg["lines_x"] = self.lines_x
        if len(self.lines_y) > 0:
            enrg["lines_y"] = self.lines_y
        if len(self.indexs_plot_lines) > 0:
            enrg["indexs_plot_lines"] = self.indexs_plot_lines
        if len(self.lines_t_x) > 0:
            enrg["lines_t_x"] = self.lines_t_x
        if len(self.lines_t_y) > 0:
            enrg["lines_t_y"] = self.lines_t_y
        if len(self.lines_t_s) > 0:
            enrg["lines_t_s"] = self.lines_t_s
        if len(self.err_x) > 0:
            enrg["err_x"] = self.err_y
        if len(self.err_y) > 0:
            enrg["err_y"] = self.err_y
        enrg["compt_color"] = self.compt_color
        if len(self.param_lines) > 0:
            enrg["param_lines"] = self.param_lines
        if len(self.param_texts) > 0:
            enrg["param_texts"] = self.param_texts
        if len(self.bords_histogramme) > 0:
            enrg["bords_histogramme"] = self.bords_histogramme
        if len(self.vals_histogramme) > 0:
            enrg["vals_histogramme"] = self.vals_histogramme
        if len(self.param_histogrammes) > 0:
            enrg["param_histogrammes"] = self.param_histogrammes
        if len(self.param_legende) > 0:
            enrg["param_legende"] = self.param_legende
        if len(self.array_image) > 1:
            enrg["array_image"] = self.array_image
            enrg["x_axe_image"] = self.x_axe_image
            enrg["y_axe_image"] = self.y_axe_image
            enrg["param_image"] = self.param_image
        if len(self.array_contours) > 1:
            enrg["array_contours"] = self.array_contours
            enrg["x_axe_contours"] = self.x_axe_contours
            enrg["y_axe_contours"] = self.y_axe_contours
        if len(self.color_label_contours) > 0:
            enrg["color_label_contours"] = self.color_label_contours
        if len(self.param_contours) > 0:
            enrg["param_contours"] = self.param_contours
        if len(self.levels) > 0:
            enrg["levels"] = self.levels
            if len(self.clabels) > 0:
                enrg["clabels"] = self.clabels
            if len(self.clabels_mask) > 0:
                enrg["clabels_mask"] = self.clabels_mask
        param = [self.title, str(self.nb_contours), str(
            int(self.tab_contours_is_image)), str(int(self.grid))]
        enrg["parameters"] = param
        if len(self.index_polygons) > 0:
            enrg["index_polygons"] = self.index_polygons
            enrg["param_polygons"] = self.param_polygons

        return enrg

    def save(self, filename: str = "graph_without_name", directory: str = None) -> None:
        """
        Save the Graphique in self.directory (default the current working directory) in npz compressed
        format.

        Parameters
        ----------

        filename: str, optinal, default="graph_without_name"
            The name of the .npz file (default: "graph_without_name")
        directory: str, optional, default="./"
            Graphique's directory (default self.directory (default : the curent working directory))

        Returns
        -------
        None

        """
        if filename != "graph_without_name":
            if ".npz" in filename:
                self.filename = filename[:-4]
            else:
                self.filename = filename
        if directory is not None:
            self.directory = directory
        enrg = dict_to_ndarray_dict(self.to_dict(), separator="-.-")
        if ".npz" not in self.filename:
            if self.directory[-1] == "/":
                np.savez_compressed(self.directory +
                                    self.filename + ".npz", **enrg)
            else:
                np.savez_compressed(self.directory + "/" +
                                    self.filename + ".npz", **enrg)
        else:
            if self.directory[-1] == "/":
                np.savez_compressed(self.directory +
                                    self.filename, **enrg)
            else:
                np.savez_compressed(self.directory + "/" +
                                    self.filename, **enrg)

    def customized_cmap(self, values: list[np.float64] | np.ndarray | tuple,
                        colors: list | np.ndarray | tuple | None = None,
                        ticks: list | np.ndarray[np.float64] | None = None,
                        **kwargs) -> None:
        """

        Parameters
        ----------

        values : list[np.float64] | np.ndarray | tuple
            The values of the colormap's color intervals if len(values)==2, the interval is automatically
            defined as a linear subdivision of the interval between values[0] and values[1] of size 255
        colors : list | np.ndarray | tuple, optional
            The associated colors, if None, a linear variation beteween C1 and C2 is bild
        ticks : list | np.ndarray[np.float64], optional
            Array of ticks for the colorbar If None, ticks are determined automatically from the input.
        kwargs
            Additional arguments for the colorbar :
                - location: str, {'right', 'top', 'bottom', 'left'}
                    Indicate where the colorbar should be plotted
                - scale: str, {'linear', 'log', 'symlog'}
                    The scale of the colorbar
                - ticks: list | array_like
                - format: str
                    ticks' format
                - label: str
                    The label to plot along the colorbar
                - size: float, default=0.01
                    relative width of the colorbar
                - fraction: float, default=1
                    relative hight of the colorbar
                - space_between: float, default=0.01
                    relative space between colorsbars (and the plot)

        Returns
        -------
        None

        Examples
        --------
        >>> x = np.linspace(0, 10, 1000)
        >>> alpha = np.linspace(1, 5, 10)
        >>> colors = linear_color_interpolation(np.arange(len(alpha)), col_min=g.C1, col_max=g.C2)
        >>> gr = Graphique()
        >>> gr.logy(x, [x**a for a in alpha], color=colors)
        >>> gr.customized_cmap(alpha, colors)
        >>> gr.show()

        """
        if len(values) < 2:
            raise UserWarning("Graphique.custumized_cmap : the len of values need to be higer than 2")
        if isinstance(values, tuple) or len(values) == 2:
            values = np.linspace(values[0], values[1], 255)
            if colors is not None and (isinstance(colors, tuple) or len(colors) == 2):
                colors = linear_color_interpolation(values, col_min=colors[0], col_max=colors[1])
        elif colors is not None and (isinstance(colors, tuple) or len(colors) == 2):
            colors = linear_color_interpolation(values, col_min=colors[0], col_max=colors[1])
        if colors is not None and len(colors) != len(values):
            raise UserWarning("Graphique.custumized_cmap : values and colors need to have the same size :"
                              "len(values)=", len(values), " len(colors)=", len(colors))
        if colors is None:
            colors = linear_color_interpolation(values, col_min=C1, col_max=C2)
        if self.custum_colorbar_values is not None:
            self.custum_colorbar_values.append(values)
        else:
            self.custum_colorbar_values = [values]
        if self.custum_colorbar_colors is not None:
            self.custum_colorbar_colors.append(colors)
        else:
            self.custum_colorbar_colors = [colors]
        if ticks is None:
            self.ticks_colorbar.append([])
        elif len(ticks) == 0:
            self.ticks_colorbar.append([-np.inf])
        else:
            self.ticks_colorbar.append(ticks)
        self.param_colorbar.append(kwargs)

    def line(self, x: np.ndarray | list, y: np.ndarray | list | None = None,
             z: np.ndarray | list | None = None,
             marker: str | list | np.ndarray[str] = "", share_colorbar: bool = True,
             scale_z: str = "linear", kwargs_colorbar: dict | None = None,
             hide: bool = False,
             axis_config: str = "bl", **kwargs) -> None:
        """

        Equivalent to plt.plot

        Parameters
        ----------

        x : array_like | list
            Abscissa(s)
        y : array_like | list, optional
            Ordinate(s), if None x became the ordinate and the abscissa is arange(len(x))
        z : array_like | list, optional
             z-axis (represented by a colorscale)
        marker : str | list[str] | array_like[str], optional, default=""
            The marker  ex ".", ",", "o", "v"... (see matplotlib documentation)
        share_colorbar : bool, optional, default=True
             If True (default) and z is not None, only one colorscale is used
             even if z is in two dimensions
        scale_z : str, {'linear', 'log', 'symlog'}, optional, default='linear'
            The scale of the z-axis
        hide : bool, optional, default=False
            If True then the new line(s) is/are not plotted with the Graphique.
            To plot them, then change the plot order with self.set_indexs_plot_lines
        kwargs_colorbar, optional
            Extra arguments for the colorbar (if z is not None)
        axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
            The positions of x-y axis :
                - "bl" is x-axis on the bottom, y-axis on the left (default).
                - "tl" is x-axis on the top, y-axis on the left.
                - "br" is x-axis on the bottom, y-axis on the right.
                - "tr" is x-axis on the top, y-axis on the right.

        kwargs, optional
            Additional argument to plot() function like linestyle, color....

        Returns
        -------
            None

        See Also
        --------
        Graphique.loglog : Graphique.line in log coordinate for boths x and y axis
        Graphique.logx : Graphique.line in log coordinate for x axis and linear for y axis
        Graphique.logy : Graphique.line in log coordinate for y axis and linear for x axis
        Graphique.point : To plot a single point
        Graphique.errorbar : To plot a line with errorbars
        Graphique.errorplot : To plot a line with errorbars represanted as filled area
        Graphique.polar : To plot a line in polar coordinates
        Graphique.symloglog : Similar to Graphique.loglog but boths negatives and positives values are represanted
        Graphique.symlogx : Similar to Graphique.logx but boths negatives and positives values are represanted
        Graphique.symlogy : Similar to Graphique.logy but boths negatives and positives values are represanted

        Notes
        -------
        This function has a small improuvment compared with plt.plot :

        if y is in two dimensions, the second dimension is plotted :
            - ```self.line(x,[y1,y2], *args)``` is equivalent to
                ```plt.plot(x, y1, *args)
                plt.plot(x, y2, *args)```
            - if y1 and y2 have not the same size:
                ```self.line([x1,x2],[y1, y2], *args)```
            - If others arguments are list of the same size of x and y, they are also split :
                ```self.line((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"])```
                is equivalent to
                ```plt.plot(x, y1, marker=".", label="Curve1")
                plt.plot(x, y2, marker=".", label="Curve2")```


        Examples
        --------
        >>> x = np.linspace(0, 10, 1000)
        >>> alpha = np.linspace(1, 5, 10)
        >>> colors = g.linear_color_interpolation(np.arange(len(alpha)), col_min=g.C1, col_max=g.C2)
        >>> gr = g.Graphique()
        >>> gr.line(x, [x*a for a in alpha], color=colors)
        >>> gr.customized_cmap(alpha, colors)
        >>> gr.show()

        """
        if not isinstance(scale_z, str):
            raise UserWarning("""Graphique.line : The scale of the z-axis need to be represanted by a string :
            'linear", 'log' or 'symlog'""")
        if axis_config not in ["bl", "tl", "tr", "br"]:
            raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)
        kwargs["axis_config"] = axis_config
        if kwargs_colorbar is None:
            kwargs_colorbar = dict()
        if not isinstance(kwargs_colorbar, dict):
            raise UserWarning("""Graphique.line : The extra arguments ir the colorbar need to be provided thru a 
            dictionary. Not a """, type(kwargs_colorbar))

        if scale_z != "linear" and scale_z != "log" and scale_z != "symlog":
            raise UserWarning("""Graphique.line : The scale of the z-axis can only be
             "linear", "log" or "symlog". Not """, scale_z)
        if isinstance(y, str):
            marker = y
            y = None
        if isinstance(z, str):
            marker = z
            z = None
        if y is None:
            y = np.copy(x)
            x = np.arange(0, len(y))
        if isinstance(x[0], list) | isinstance(x[0], np.ndarray):
            if isinstance(x[0][0], list) | isinstance(x[0][0], np.ndarray):
                raise UserWarning("Graphique.line the x-axis dimension cannot be superior than 2")
            else:
                dim_x: int = 2
        else:
            dim_x: int = 1
        if isinstance(y[0], list) | isinstance(y[0], np.ndarray):
            if isinstance(y[0][0], list) | isinstance(y[0][0], np.ndarray):
                raise UserWarning("Graphique.line the y-axis dimension cannot be superior than 2")
            else:
                dim_y: int = 2
        else:
            dim_y: int = 1
        if z is not None and isinstance(z[0], list) | isinstance(z[0], np.ndarray):
            if isinstance(z[0][0], list) | isinstance(z[0][0], np.ndarray):
                raise UserWarning("Graphique.line the z-axis dimension cannot be superior than 2")
            else:
                dim_z: int = 2
        elif z is None:
            dim_z: int = 0
        else:
            dim_z: int = 1
        if (dim_x == 2 and dim_y == 2 and
                np.any(np.array([len(X) != len(Y) for (X, Y) in zip(x, y)]))):
            raise UserWarning("Graphique.line : the sizes of arrays of the abscissa "
                              "doesn't mach with the sizes of the array of ordinates : ",
                              [(len(X), len(Y)) for (X, Y) in zip(x, y)])
        elif (dim_x == 2 and dim_y == 2 and dim_z == 2 and
              np.any(np.array([len(Y) != len(Z) for (Y, Z) in zip(y, z)]))):
            raise UserWarning("Graphique.line : the sizes of arrays of the z-axis "
                              "doesn't mach with the sizes of the array of ordinates : ",
                              [(len(Y), len(Z)) for (Y, Z) in zip(y, z)])
        elif (dim_x == 2 and dim_y == 2 and dim_z == 1 and
              (np.any(np.array([len(Y) != len(z) for Y in y]))) and len(y) != len(z)):
            raise UserWarning("Graphique.line : the sizes of the z-axis "
                              "doesn't mach with the sizes of the array of ordinates : ",
                              [(len(Y), len(Z)) for (Y, Z) in zip(y, z)], (len(y), len(z)))
        elif (dim_y == 2 and dim_x == 1 and
              np.any(np.array([len(x) != len(Y) for Y in y]))):
            raise UserWarning("Graphique.line : the sizes of the abscissa "
                              "doesn't mach with the sizes of the array of ordinates : ",
                              [(len(x), len(Y)) for Y in y])
        elif (dim_y == 2 and dim_x == 1 and dim_z == 2 and
              (np.any(np.array([len(x) != len(Z) for Z in z])) and len(y) != len(z))):
            raise UserWarning("Graphique.line : the sizes of the array of z-axis "
                              "doesn't mach with the sizes of the abscissa : ",
                              [(len(x), len(Z)) for Z in z], " or the y-axis :", (len(y), len(z)))
        elif (dim_y == 2 and dim_x == 1 and dim_z == 1 and
              len(x) != len(z) and len(y) != len(z)):
            raise UserWarning("Graphique.line : the sizes of thez-axis "
                              "doesn't mach with the sizes of the abscissa : ",
                              len(x), len(z), " or the y-axis :", len(y))
        elif dim_y == 1 and dim_x == 1 and len(x) != len(y):
            raise UserWarning("Graphique.line : the sizes of the abscissa "
                              "doesn't mach with the sizes of the ordinate : ",
                              len(x), len(y))
        elif dim_y == 1 and dim_x == 1 and dim_z == 2:
            raise UserWarning("Graphique.line : There is only one list of ordinate for several's lists"
                              "of z. This cannot be represented by a colorscale")
        elif dim_y == 1 and dim_x == 1 and dim_z == 1 and len(x) != len(z):
            raise UserWarning("Graphique.line : the sizes of the abscissa "
                              "doesn't mach with the sizes of the z-axis : ",
                              len(x), len(z))

        if dim_x == 2 and dim_y == 2 and dim_z == 2:
            if scale_z == "linear" or scale_z == "symlog":
                z_min: np.float64 = np.min(z)
                z_max: np.float64 = np.max(z)
            else:
                if z is not None and np.any(z > 0):
                    z_min: np.float64 = np.min(z[z > 0])
                    z_max: np.float64 = np.max(z)
                else:
                    raise UserWarning("Graphique.line : z-axis has no strictly positive values and the scale is log)")

            if scale_z == "symlog" and z is not None and (z > 0).sum() > 0:
                z_min_pos: np.float64 = np.min(z[z > 0])
            else:
                z_min_pos: np.float64 = 0.
            if scale_z == "symlog" and z is not None and (z < 0).sum() > 0:
                z_max_neg: np.float64 = np.max(z[z < 0])
            else:
                z_max_neg: np.float64 = 0.

            locs: list[str] = ["right", "top", "left"]
            fracs: list[str] = [0.15, 0.05, 0.15]
            aspects: list[str] = [20, 30, 20]
            for (X, Y, Z, i) in zip(x, y, z, np.arange(len(x))):
                if not share_colorbar:
                    if scale_z == "linear" or scale_z == "symlog":
                        z_min: np.float64 = np.min(Z)
                        z_max: np.float64 = np.max(Z)
                    else:
                        if np.any(z > 0):
                            z_min: np.float64 = np.min(Z[Z > 0])
                            z_max: np.float64 = np.max(Z)
                        else:
                            z_min: np.float64 = np.double(0.)
                            z_max: np.float64 = np.double(0.)
                    if scale_z == "symlog" and (Z > 0).sum() > 0:
                        z_min_pos: np.float64 = np.min(Z[Z > 0])
                    else:
                        z_min_pos: np.float64 = 0.
                    if scale_z == "symlog" and (Z < 0).sum() > 0:
                        z_max_neg: np.float64 = np.max(Z[Z < 0])
                    else:
                        z_max_neg: np.float64 = 0
                idx_s: np.ndarray[int] = np.arange(len(Z))
                if scale_z == "log" and np.any(Z > 0):
                    idx_s = np.argwhere(Z > 0)[:, 0]
                elif scale_z == "log":
                    idx_s = np.array([])
                if len(idx_s) > 0:
                    self.lines_x.append(np.array(X[idx_s]))
                    self.lines_y.append(np.array(Y[idx_s]))
                else:
                    self.lines_x.append(np.array([]))
                    self.lines_y.append(np.array([]))
                self.err_x.append([])
                self.err_y.append([])
                args_auxi: dict = {}
                for k in kwargs.keys():
                    if (isinstance(kwargs[k], list) | isinstance(kwargs[k], np.ndarray)) and len(kwargs[k]) == len(y):
                        args_auxi[k] = kwargs[k][i]
                    else:
                        args_auxi[k] = kwargs[k]
                if marker != "" and not ("linestyle" in args_auxi):
                    args_auxi["linestyle"] = ""
                    if ((isinstance(marker, list) | isinstance(marker, np.ndarray))
                            and len(marker) == len(y)):
                        args_auxi["marker"] = marker[i]
                    else:
                        args_auxi["marker"] = marker
                elif marker != "":
                    if ((isinstance(marker, list) | isinstance(marker, np.ndarray))
                            and len(marker) == len(y)):
                        args_auxi["marker"] = marker[i]
                    else:
                        args_auxi["marker"] = marker
                if scale_z == "linear":
                    if share_colorbar:
                        c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                        c_max: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                    else:
                        c_min: str = l_colors[(self.compt_color + 1 + i) % len(l_colors)]
                        c_max: str = l_colors[(self.compt_color + 2 + i) % len(l_colors)]

                    args_auxi["color"] = linear_color_interpolation(Z, val_min=z_min, val_max=z_max,
                                                                    col_min=c_min,
                                                                    col_max=c_max)
                    if not share_colorbar:
                        self.customized_cmap(values=np.linspace(z_min, z_max, 255),
                                             colors=linear_color_interpolation(np.linspace(z_min, z_max, 255),
                                                                               col_min=c_min, col_max=c_max),
                                             location=locs[i % len(locs)],
                                             fraction=fracs[i % len(fracs)],
                                             aspect=aspects[i % len(aspects)], scale=scale_z, **kwargs_colorbar)
                elif scale_z == "log" and len(idx_s) > 0:
                    if share_colorbar:
                        c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                        c_max: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                    else:
                        c_min: str = l_colors[(self.compt_color + 1 + i) % len(l_colors)]
                        c_max: str = l_colors[(self.compt_color + 2 + i) % len(l_colors)]

                    args_auxi["color"] = linear_color_interpolation(np.log10(Z[idx_s]), val_min=np.log10(z_min),
                                                                    val_max=np.log10(z_max),
                                                                    col_min=c_min,
                                                                    col_max=c_max)
                    if not share_colorbar:
                        self.customized_cmap(values=np.linspace(np.log10(z_min), np.log10(z_max), 255),
                                             colors=linear_color_interpolation(np.linspace(z_min, z_max, 255),
                                                                               col_min=c_min, col_max=c_max),
                                             location=locs[i % len(locs)],
                                             fraction=fracs[i % len(fracs)],
                                             aspect=aspects[i % len(aspects)], scale=scale_z, **kwargs_colorbar)
                else:
                    if share_colorbar:
                        c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                        c_med: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                        c_max: str = l_colors[(self.compt_color + 3) % len(l_colors)]
                    else:
                        c_min: str = l_colors[(self.compt_color + 1 + i) % len(l_colors)]
                        c_med: str = l_colors[(self.compt_color + 2 + i) % len(l_colors)]
                        c_max: str = l_colors[(self.compt_color + 3 + i) % len(l_colors)]
                    ma: np.ndarray[bool] = Z > 0.
                    args_auxi["color"] = np.full(len(X), c_med)
                    if z_min_pos > 0.:
                        args_auxi["color"][ma] = linear_color_interpolation(np.log10(Z[ma]),
                                                                            val_min=np.log10(z_min_pos),
                                                                            val_max=np.log10(z_max),
                                                                            col_min=c_med, col_max=c_max)
                        z_colors1 = np.geomspace(z_min_pos, z_max, 255 // 2)
                        colors1 = linear_color_interpolation(np.linspace(np.log10(z_min_pos), np.log10(z_max),
                                                                         255 // 2),
                                                             col_min=c_med, col_max=c_max)
                    else:
                        z_colors1 = np.array([])
                        colors1 = np.array([])

                    if z_max_neg < 0.:
                        args_auxi["color"][~ma] = linear_color_interpolation(np.log10(-Z[~ma]),
                                                                             val_min=np.log10(-z_max_neg),
                                                                             val_max=np.log10(-z_min),
                                                                             col_min=c_med, col_max=c_min)
                        z_colors2 = -np.geomspace(-z_min, -z_max_neg, 255 // 2)
                        colors2 = linear_color_interpolation(np.linspace(np.log10(-z_min), np.log10(-z_max_neg),
                                                                         255 // 2),
                                                             col_min=c_med, col_max=c_min)
                    else:
                        z_colors2 = np.array([])
                        colors2 = np.array([])
                    z_colors: np.ndarray[np.float64] = np.append(z_colors2, z_colors1)
                    colors: np.ndarray[np.float64] = np.append(colors2, colors1)
                    if not share_colorbar:
                        self.customized_cmap(values=z_colors,
                                             colors=colors,
                                             location=locs[i % len(locs)],
                                             fraction=fracs[i % len(fracs)],
                                             aspect=aspects[i % len(aspects)], scale=scale_z, **kwargs_colorbar)
                if "label" in kwargs and args_auxi["label"] == "":
                    del args_auxi["label"]
                    # Delete empty legend to prevent a warning message and the addition of an empty gray square
                self.param_lines.append(args_auxi)
            if share_colorbar:
                if scale_z == "linear":
                    c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                    c_max: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                    self.customized_cmap(values=np.linspace(z_min, z_max, 255),
                                         colors=linear_color_interpolation(np.linspace(z_min, z_max, 255),
                                                                           col_min=c_min, col_max=c_max),
                                         scale=scale_z, **kwargs_colorbar)
                    self.compt_color += 2
                elif scale_z == "log":
                    c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                    c_max: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                    self.customized_cmap(values=np.geomspace(z_min, z_max, 255),
                                         colors=linear_color_interpolation(np.linspace(z_min, z_max, 255),
                                                                           col_min=c_min, col_max=c_max),
                                         scale=scale_z, **kwargs_colorbar)
                    self.compt_color += 2
                else:
                    c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                    c_med: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                    c_max: str = l_colors[(self.compt_color + 3) % len(l_colors)]
                    if z_min_pos > 0.:
                        z_colors1 = np.geomspace(z_min_pos, z_max, 255 // 2)
                        colors1 = linear_color_interpolation(np.linspace(np.log10(z_min_pos), np.log10(z_max),
                                                                         255 // 2),
                                                             col_min=c_med, col_max=c_max)
                    else:
                        z_colors1 = np.array([])
                        colors1 = np.array([])
                    if z_max_neg < 0.:
                        z_colors2 = -np.geomspace(-z_min, -z_max_neg, 255 // 2)
                        colors2 = linear_color_interpolation(np.linspace(np.log10(-z_min), np.log10(-z_max_neg),
                                                                         255 // 2),
                                                             col_min=c_med, col_max=c_min)
                    else:
                        z_colors2 = np.array([])
                        colors2 = np.array([])
                    z_colors: np.ndarray[np.float64] = np.append(z_colors2, z_colors1)
                    colors: np.ndarray[np.float64] = np.append(colors2, colors1)
                    self.customized_cmap(values=z_colors, colors=colors, scale=scale_z, **kwargs_colorbar)
                    self.compt_color += 3
            elif scale_z == "linear" or scale_z == "log":
                self.compt_color += 2 * len(x)
            else:
                self.compt_color += 3 * len(x)

        elif dim_x == 2 and dim_y == 2 and dim_z < 2:
            if z is not None and scale_z == "linear" or scale_z == "symlog":
                z_min: np.float64 = np.min(z)
                z_max: np.float64 = np.max(z)
                c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                c_max: str = l_colors[(self.compt_color + 2) % len(l_colors)]
            elif z is not None:
                c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                c_med: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                c_max: str = l_colors[(self.compt_color + 3) % len(l_colors)]
                if np.any(z > 0):
                    z_min: np.float64 = np.min(z[z > 0])
                    z_max: np.float64 = np.max(z)
                else:
                    raise UserWarning("Graphique.line : z-axis has no strictly positive values and the scale is log)")
            else:
                z_min: np.float64 = np.double(0.)
                z_max: np.float64 = np.double(0.)
            if z is not None and scale_z == "symlog" and (z > 0).sum() > 0:
                z_min_pos: np.float64 = np.min(z[z > 0])
            else:
                z_min_pos: np.float64 = 0.
            if z is not None and scale_z == "symlog" and (z < 0).sum() > 0:
                z_max_neg: np.float64 = np.max(z[z < 0])
            else:
                z_max_neg: np.float64 = 0.

            idx_s: np.ndarray[int] = np.arange(len(x[0]))
            if scale_z == "log" and np.any(z > 0):
                idx_s = np.argwhere(z > 0)[:, 0]
            elif scale_z == "log":
                idx_s = np.array([])
            colors: np.ndarray[str] = []
            if scale_z == "linear" and z is not None:
                colors = linear_color_interpolation(z, val_min=z_min, val_max=z_max,
                                                    col_min=c_min, col_max=c_max)
            elif scale_z == "log" and z is not None and np.any(z > 0):
                colors = linear_color_interpolation(np.log10(z[idx_s]), val_min=np.log10(z_min),
                                                    val_max=np.log10(z_max), col_min=c_min, col_max=c_max)
            elif z is not None:
                colors = np.full(len(z), c_med)
                ma = z > 0
                if z_min_pos > 0.:
                    colors[ma] = linear_color_interpolation(np.log10(colors[ma]),
                                                            val_min=np.log10(z_min_pos),
                                                            val_max=np.log10(z_max),
                                                            col_min=c_med, col_max=c_max)
                else:
                    z_colors1 = np.array([])
                    colors1 = np.array([])

                if z_max_neg < 0.:
                    colors[~ma] = linear_color_interpolation(np.log10(-z[~ma]),
                                                             val_min=np.log10(-z_max_neg),
                                                             val_max=np.log10(-z_min),
                                                             col_min=c_med, col_max=c_min)

            for (X, Y, i) in zip(x, y, np.arange(len(x))):
                if not (z is not None and len(z) == len(y) and i not in idx_s):
                    if z is not None and len(z) == len(Y):
                        self.lines_x.append(np.array(X[idx_s]))
                        self.lines_y.append(np.array(Y[idx_s]))
                    else:
                        self.lines_x.append(np.array(X))
                        self.lines_y.append(np.array(Y))
                    self.err_x.append([])
                    self.err_y.append([])
                    args_auxi: dict = {}

                    for k in kwargs.keys():
                        if (isinstance(kwargs[k], list) | isinstance(kwargs[k], np.ndarray)) and len(kwargs[k]) == len(
                                y):
                            args_auxi[k] = kwargs[k][i]
                        else:
                            args_auxi[k] = kwargs[k]
                    if marker != "" and not ("linestyle" in args_auxi):
                        args_auxi["linestyle"] = ""
                        if ((isinstance(marker, list) | isinstance(marker, np.ndarray))
                                and len(marker) == len(y)):
                            args_auxi["marker"] = marker[i]
                        else:
                            args_auxi["marker"] = marker
                    elif marker != "":
                        if ((isinstance(marker, list) | isinstance(marker, np.ndarray))
                                and len(marker) == len(y)):
                            args_auxi["marker"] = marker[i]
                        else:
                            args_auxi["marker"] = marker
                    if z is None:
                        if "color" not in kwargs and len(y) <= 4:
                            args_auxi["color"] = l_colors[(self.compt_color + 1 + i) % len(l_colors)]
                        elif "color" not in kwargs:
                            args_auxi["color"] = linear_color_interpolation(i, val_min=0, val_max=len(y) - 1,
                                                                            col_min=l_colors[(self.compt_color + 1)
                                                                                             % len(l_colors)],
                                                                            col_max=l_colors[(self.compt_color + 2)
                                                                                             % len(l_colors)])

                    elif len(z) == len(y):
                        args_auxi["color"] = colors[i]
                    else:
                        args_auxi["color"] = colors
                    if "label" in kwargs and args_auxi["label"] == "":
                        del args_auxi["label"]
                        # Delete empty legend to prevent a warning message and the addition of an empty gray square
                    self.param_lines.append(args_auxi)

            if z is None and len(y) <= 4 and "color" not in kwargs:
                self.compt_color += len(y)
            elif z is None and "color" not in kwargs:
                self.compt_color += 1
            elif scale_z == "linear":
                self.customized_cmap(values=np.linspace(z_min, z_max, 255),
                                     colors=linear_color_interpolation(np.linspace(z_min, z_max, 255),
                                                                       col_min=c_min, col_max=c_max),
                                     scale=scale_z, **kwargs_colorbar)
                self.compt_color += 2
            elif scale_z == "log":
                self.customized_cmap(values=np.geomspace(z_min, z_max, 255),
                                     colors=linear_color_interpolation(np.linspace(z_min, z_max, 255),
                                                                       col_min=c_min, col_max=c_max),
                                     scale=scale_z, **kwargs_colorbar)
                self.compt_color += 2
            else:
                if z_min_pos > 0.:
                    z_colors1 = np.geomspace(z_min_pos, z_max, 255 // 2)
                    colors1 = linear_color_interpolation(np.linspace(np.log10(z_min_pos), np.log10(z_max),
                                                                     255 // 2),
                                                         col_min=c_med, col_max=c_max)
                else:
                    z_colors1 = np.array([])
                    colors1 = np.array([])
                if z_max_neg < 0.:
                    z_colors2 = -np.geomspace(-z_min, -z_max_neg, 255 // 2)
                    colors2 = linear_color_interpolation(np.linspace(np.log10(-z_min), np.log10(-z_max_neg),
                                                                     255 // 2),
                                                         col_min=c_med, col_max=c_min)
                else:
                    z_colors2 = np.array([])
                    colors2 = np.array([])
                z_colors: np.ndarray[np.float64] = np.append(z_colors2, z_colors1)
                colors: np.ndarray[np.float64] = np.append(colors2, colors1)
                self.customized_cmap(values=z_colors, colors=colors, scale=scale_z, **kwargs_colorbar)
                self.compt_color += 3

        elif dim_y == 2 and dim_z == 2:
            if scale_z == "linear" or scale_z == "symlog":
                z_min: np.float64 = np.min(z)
                z_max: np.float64 = np.max(z)
            else:
                if np.any(z > 0):
                    z_min: np.float64 = np.min(z[z > 0])
                    z_max: np.float64 = np.max(z)
                else:
                    raise UserWarning("Graphique.line : z-axis has no strictly positive values and the scale is log)")

            if scale_z == "symlog" and (z > 0).sum() > 0:
                z_min_pos: np.float64 = np.min(z[z > 0])
            else:
                z_min_pos: np.float64 = 0.
            if scale_z == "symlog" and z is not None and (z < 0).sum() > 0:
                z_max_neg: np.float64 = np.max(z[z < 0])
            else:
                z_max_neg: np.float64 = 0.

            locs: list[str] = ["right", "top", "left"]
            fracs: list[str] = [0.15, 0.05, 0.15]
            aspects: list[str] = [20, 30, 20]
            for (Y, Z, i) in zip(y, z, np.arange(len(y))):
                if not share_colorbar:
                    if scale_z == "linear" or scale_z == "symlog":
                        z_min: np.float64 = np.min(Z)
                        z_max: np.float64 = np.max(Z)
                    else:
                        if np.any(z > 0):
                            z_min: np.float64 = np.min(Z[Z > 0])
                            z_max: np.float64 = np.max(Z)
                        else:
                            z_min: np.float64 = np.double(0.)
                            z_max: np.float64 = np.double(0.)
                    if scale_z == "symlog" and (Z > 0).sum() > 0:
                        z_min_pos: np.float64 = np.min(Z[Z > 0])
                    else:
                        z_min_pos: np.float64 = 0.
                    if scale_z == "symlog" and (Z < 0).sum() > 0:
                        z_max_neg: np.float64 = np.max(Z[Z < 0])
                    else:
                        z_max_neg: np.float64 = 0
                if scale_z == "log":
                    idx_s = np.argwhere(Z > 0)[:, 0]

                idx_s: np.ndarray[int] = np.arange(len(Z))
                if scale_z == "log" and np.any(Z > 0):
                    idx_s = np.argwhere(Z > 0)[:, 0]
                elif scale_z == "log":
                    idx_s = np.array([])
                if len(idx_s) > 0:
                    self.lines_x.append(np.array(x[idx_s]))
                    self.lines_y.append(np.array(Y[idx_s]))
                else:
                    self.lines_x.append(np.array([]))
                    self.lines_y.append(np.array([]))
                self.err_x.append([])
                self.err_y.append([])
                args_auxi: dict = {}
                for k in kwargs.keys():
                    if (isinstance(kwargs[k], list) | isinstance(kwargs[k], np.ndarray)) and len(kwargs[k]) == len(y):
                        args_auxi[k] = kwargs[k][i]
                    else:
                        args_auxi[k] = kwargs[k]
                if marker != "" and not ("linestyle" in args_auxi):
                    args_auxi["linestyle"] = ""
                    if ((isinstance(marker, list) | isinstance(marker, np.ndarray))
                            and len(marker) == len(y)):
                        args_auxi["marker"] = marker[i]
                    else:
                        args_auxi["marker"] = marker
                elif marker != "":
                    if ((isinstance(marker, list) | isinstance(marker, np.ndarray))
                            and len(marker) == len(y)):
                        args_auxi["marker"] = marker[i]
                    else:
                        args_auxi["marker"] = marker
                if scale_z == "linear":
                    if share_colorbar:
                        c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                        c_max: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                    else:
                        c_min: str = l_colors[(self.compt_color + 1 + i) % len(l_colors)]
                        c_max: str = l_colors[(self.compt_color + 2 + i) % len(l_colors)]

                    args_auxi["color"] = linear_color_interpolation(Z, val_min=z_min, val_max=z_max,
                                                                    col_min=c_min,
                                                                    col_max=c_max)
                    if not share_colorbar:
                        self.customized_cmap(values=np.linspace(z_min, z_max, 255),
                                             colors=linear_color_interpolation(np.linspace(z_min, z_max, 255),
                                                                               col_min=c_min, col_max=c_max),
                                             location=locs[i % len(locs)],
                                             fraction=fracs[i % len(fracs)],
                                             aspect=aspects[i % len(aspects)],
                                             scale=scale_z, **kwargs_colorbar)
                elif scale_z == "log" and len(idx_s) > 0:
                    if share_colorbar:
                        c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                        c_max: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                    else:
                        c_min: str = l_colors[(self.compt_color + 1 + i) % len(l_colors)]
                        c_max: str = l_colors[(self.compt_color + 2 + i) % len(l_colors)]

                    args_auxi["color"] = linear_color_interpolation(np.log10(Z[idx_s]), val_min=np.log10(z_min),
                                                                    val_max=np.log10(z_max),
                                                                    col_min=c_min,
                                                                    col_max=c_max)
                    if not share_colorbar:
                        self.customized_cmap(values=np.geomspace(z_min, z_max, 255),
                                             colors=linear_color_interpolation(np.linspace(z_min, z_max, 255),
                                                                               col_min=c_min, col_max=c_max),
                                             location=locs[i % len(locs)],
                                             fraction=fracs[i % len(fracs)],
                                             aspect=aspects[i % len(aspects)],
                                             scale=scale_z, **kwargs_colorbar)
                else:
                    if share_colorbar:
                        c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                        c_med: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                        c_max: str = l_colors[(self.compt_color + 3) % len(l_colors)]
                    else:
                        c_min: str = l_colors[(self.compt_color + 1 + i) % len(l_colors)]
                        c_med: str = l_colors[(self.compt_color + 2 + i) % len(l_colors)]
                        c_max: str = l_colors[(self.compt_color + 3 + i) % len(l_colors)]
                    ma: np.ndarray[bool] = Z > 0.
                    args_auxi["color"] = np.full(len(x), c_med)
                    if z_min_pos > 0.:
                        args_auxi["color"][ma] = linear_color_interpolation(np.log10(Z[ma]),
                                                                            val_min=np.log10(z_min_pos),
                                                                            val_max=np.log10(z_max),
                                                                            col_min=c_med, col_max=c_max)
                        z_colors1 = np.geomspace(z_min_pos, z_max, 255 // 2)
                        colors1 = linear_color_interpolation(np.linspace(np.log10(z_min_pos), np.log10(z_max),
                                                                         255 // 2),
                                                             col_min=c_med, col_max=c_max)
                    else:
                        z_colors1 = np.array([])
                        colors1 = np.array([])

                    if z_max_neg < 0.:
                        args_auxi["color"][~ma] = linear_color_interpolation(np.log10(-Z[~ma]),
                                                                             val_min=np.log10(-z_max_neg),
                                                                             val_max=np.log10(-z_min),
                                                                             col_min=c_med, col_max=c_min)
                        z_colors2 = -np.geomspace(-z_min, -z_max_neg, 255 // 2)
                        colors2 = linear_color_interpolation(np.linspace(np.log10(-z_min), np.log10(-z_max_neg),
                                                                         255 // 2),
                                                             col_min=c_med, col_max=c_min)
                    else:
                        z_colors2 = np.array([])
                        colors2 = np.array([])
                    z_colors: np.ndarray[np.float64] = np.append(z_colors2, z_colors1)
                    colors: np.ndarray[np.float64] = np.append(colors2, colors1)
                    if not share_colorbar:
                        self.customized_cmap(values=z_colors,
                                             colors=colors,
                                             location=locs[i % len(locs)],
                                             fraction=fracs[i % len(fracs)],
                                             aspect=aspects[i % len(aspects)], scale=scale_z,
                                             **kwargs_colorbar)
                if "label" in kwargs and args_auxi["label"] == "":
                    del args_auxi["label"]
                    # Delete empty legend to prevent a warning message and the addition of an empty gray square
                self.param_lines.append(args_auxi)
            if share_colorbar:
                if scale_z == "linear":
                    c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                    c_max: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                    self.customized_cmap(values=np.linspace(z_min, z_max, 255),
                                         colors=linear_color_interpolation(np.linspace(z_min, z_max, 255),
                                                                           col_min=c_min, col_max=c_max),
                                         scale=scale_z, **kwargs_colorbar)
                    self.compt_color += 2
                elif scale_z == "log":
                    c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                    c_max: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                    self.customized_cmap(values=np.geomspace(z_min, z_max, 255),
                                         colors=linear_color_interpolation(np.linspace(z_min, z_max, 255),
                                                                           col_min=c_min, col_max=c_max),
                                         scale=scale_z, **kwargs_colorbar)
                    self.compt_color += 2
                else:
                    c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                    c_med: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                    c_max: str = l_colors[(self.compt_color + 3) % len(l_colors)]
                    if z_min_pos > 0.:
                        z_colors1 = np.geomspace(z_min_pos, z_max, 255 // 2)
                        colors1 = linear_color_interpolation(np.linspace(np.log10(z_min_pos), np.log10(z_max),
                                                                         255 // 2),
                                                             col_min=c_med, col_max=c_max)
                    else:
                        z_colors1 = np.array([])
                        colors1 = np.array([])
                    if z_max_neg < 0.:
                        z_colors2 = -np.geomspace(-z_min, -z_max_neg, 255 // 2)
                        colors2 = linear_color_interpolation(np.linspace(np.log10(-z_min), np.log10(-z_max_neg),
                                                                         255 // 2),
                                                             col_min=c_med, col_max=c_min)
                    else:
                        z_colors2 = np.array([])
                        colors2 = np.array([])
                    z_colors: np.ndarray[np.float64] = np.append(z_colors2, z_colors1)
                    colors: np.ndarray[np.float64] = np.append(colors2, colors1)
                    self.customized_cmap(values=z_colors, colors=colors, scale=scale_z, **kwargs_colorbar)
                    self.compt_color += 3
            elif scale_z == "linear" or scale_z == "log":
                self.compt_color += 2 * len(x)
            else:
                self.compt_color += 3 * len(x)

        elif dim_y == 2 and dim_z < 2:
            if z is not None and scale_z == "linear" or scale_z == "log":
                c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                c_max: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                if scale_z == "log" and np.any(z > 0):
                    z_min: np.float64 = np.min(z[z > 0])
                    z_max: np.float64 = np.max(z)
                elif scale_z == "log":
                    raise UserWarning("Graphique.line : z-axis has no strictly positive values and the scale is log)")
                else:
                    z_min: np.float64 = np.min(z)
                    z_max: np.float64 = np.max(z)
            elif z is not None:
                c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                c_med: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                c_max: str = l_colors[(self.compt_color + 3) % len(l_colors)]
            else:
                z_min: np.float64 = np.double(0.)
                z_max: np.float64 = np.double(0.)
            if z is not None and scale_z == "symlog" and np.any(z > 0.):
                z_min_pos: np.float64 = np.min(z[z > 0])
            else:
                z_min_pos: np.float64 = 0.
            if z is not None and scale_z == "symlog" and np.any(z > 0.):
                z_max_neg: np.float64 = np.max(z[z < 0])
            else:
                z_max_neg: np.float64 = 0.
            colors: np.ndarray[str] = []
            if scale_z == "linear" and z is not None:
                colors = linear_color_interpolation(z, val_min=z_min, val_max=z_max,
                                                    col_min=c_min, col_max=c_max)
            elif scale_z == "log" and z is not None and np.any(z > 0):
                colors = linear_color_interpolation(np.log10(z[z > 0]), val_min=np.log10(z_min),
                                                    val_max=np.log10(z_max), col_min=c_min, col_max=c_max)
            else:
                if z is not None:
                    colors = np.full(len(z), c_med)
                    ma = z > 0
                    if z_min_pos > 0.:
                        colors[ma] = linear_color_interpolation(np.log10(colors[ma]),
                                                                val_min=np.log10(z_min_pos),
                                                                val_max=np.log10(z_max),
                                                                col_min=c_med, col_max=c_max)
                    else:
                        z_colors1 = np.array([])
                        colors1 = np.array([])

                    if z_max_neg < 0.:
                        colors[~ma] = linear_color_interpolation(np.log10(-z[~ma]),
                                                                 val_min=np.log10(-z_max_neg),
                                                                 val_max=np.log10(-z_min),
                                                                 col_min=c_med, col_max=c_min)

            idx_s: np.ndarray[int] = np.arange(len(x))
            if z is not None and scale_z == "log" and np.any(z > 0):
                idx_s = np.argwhere(z > 0)[:, 0]
            elif z is not None and scale_z == "log":
                idx_s = np.array([])
            for (Y, i) in zip(y, np.arange(len(y))):
                if not (z is not None and len(z) == len(y) and i not in idx_s):
                    if z is not None and len(z) == len(Y):
                        self.lines_x.append(np.array(x)[idx_s])
                        self.lines_y.append(np.array(Y)[idx_s])
                    else:
                        self.lines_x.append(np.array(x))
                        self.lines_y.append(np.array(Y))
                    self.err_x.append([])
                    self.err_y.append([])
                    args_auxi: dict = {}

                    for k in kwargs.keys():
                        if (isinstance(kwargs[k], list) | isinstance(kwargs[k], np.ndarray)) and len(kwargs[k]) == len(
                                y):
                            args_auxi[k] = kwargs[k][i]
                        else:
                            args_auxi[k] = kwargs[k]
                    if marker != "" and not ("linestyle" in args_auxi):
                        args_auxi["linestyle"] = ""
                        if ((isinstance(marker, list) | isinstance(marker, np.ndarray))
                                and len(marker) == len(y)):
                            args_auxi["marker"] = marker[i]
                        else:
                            args_auxi["marker"] = marker
                    elif marker != "":
                        if ((isinstance(marker, list) | isinstance(marker, np.ndarray))
                                and len(marker) == len(y)):
                            args_auxi["marker"] = marker[i]
                        else:
                            args_auxi["marker"] = marker
                    if z is None:
                        if "color" not in kwargs and len(y) <= 4:
                            args_auxi["color"] = l_colors[(self.compt_color + 1 + i) % len(l_colors)]
                        elif "color" not in kwargs:
                            args_auxi["color"] = linear_color_interpolation(i, val_min=0, val_max=len(y) - 1,
                                                                            col_min=l_colors[(self.compt_color + 1)
                                                                                             % len(l_colors)],
                                                                            col_max=l_colors[(self.compt_color + 2)
                                                                                             % len(l_colors)])

                    elif len(z) == len(y):
                        args_auxi["color"] = colors[i]
                    else:
                        args_auxi["color"] = colors
                    if "label" in kwargs and args_auxi["label"] == "":
                        del args_auxi["label"]
                        # Delete empty legend to prevent a warning message and the addition of an empty gray square
                    self.param_lines.append(args_auxi)

            if z is None and len(y) <= 4 and "color" not in kwargs:
                self.compt_color += len(y)
            elif z is None and "color" not in kwargs:
                self.compt_color += 2
            elif z is not None and scale_z == "linear":
                self.customized_cmap(values=np.linspace(z_min, z_max, 255),
                                     colors=linear_color_interpolation(np.linspace(z_min, z_max, 255),
                                                                       col_min=c_min, col_max=c_max),
                                     scale=scale_z, **kwargs_colorbar)
                self.compt_color += 2
            elif z is not None and scale_z == "log":
                self.customized_cmap(values=np.geomspace(z_min, z_max, 255),
                                     colors=linear_color_interpolation(np.linspace(z_min, z_max, 255),
                                                                       col_min=c_min, col_max=c_max),
                                     scale=scale_z, **kwargs_colorbar)
                self.compt_color += 2
            elif z is not None:
                if z_min_pos > 0.:
                    z_colors1 = np.geomspace(z_min_pos, z_max, 255 // 2)
                    colors1 = linear_color_interpolation(np.linspace(np.log10(z_min_pos), np.log10(z_max),
                                                                     255 // 2),
                                                         col_min=c_med, col_max=c_max)
                else:
                    z_colors1 = np.array([])
                    colors1 = np.array([])
                if z_max_neg < 0.:
                    z_colors2 = -np.geomspace(-z_min, -z_max_neg, 255 // 2)
                    colors2 = linear_color_interpolation(np.linspace(np.log10(-z_min), np.log10(-z_max_neg),
                                                                     255 // 2),
                                                         col_min=c_med, col_max=c_min)
                else:
                    z_colors2 = np.array([])
                    colors2 = np.array([])
                z_colors: np.ndarray[np.float64] = np.append(z_colors2, z_colors1)
                colors: np.ndarray[np.float64] = np.append(colors2, colors1)
                self.customized_cmap(values=z_colors, colors=colors, scale=scale_z, **kwargs_colorbar)
                self.compt_color += 3
        else:
            idx_s: np.ndarray[int] = np.arange(len(x))
            if scale_z == "log" and np.any(z > 0):
                idx_s = np.argwhere(z > 0)[:, 0]
            elif scale_z == "log":
                idx_s = np.array([])
            self.lines_x.append(np.array(x)[idx_s])
            self.lines_y.append(np.array(y)[idx_s])
            self.err_x.append([])
            self.err_y.append([])
            if marker != "" and not ("linestyle" in kwargs):
                kwargs["linestyle"] = ""
                kwargs["marker"] = marker
            elif marker != "":
                kwargs["marker"] = marker
            if z is None and "color" not in kwargs:
                kwargs["color"] = l_colors[self.compt_color + 1 % len(l_colors)]
                self.compt_color += 1
            elif z is not None:
                if scale_z == "linear":
                    c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                    c_max: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                    z_min: np.float64 = np.min(z)
                    z_max: np.float64 = np.max(z)
                    kwargs["color"] = linear_color_interpolation(z, val_min=z_min, val_max=z_max,
                                                                 col_min=c_min, col_max=c_max)
                    self.customized_cmap(values=np.linspace(z_min, z_max, 255),
                                         colors=linear_color_interpolation(np.linspace(z_min, z_max, 255),
                                                                           col_min=c_min, col_max=c_max),
                                         scale=scale_z, **kwargs_colorbar)
                    self.compt_color += 2
                elif scale_z == "log" and np.any(z > 0.):
                    c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                    c_max: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                    z_min: np.float64 = np.min(z[z > 0])
                    z_max: np.float64 = np.max(z[z > 0])

                    kwargs["color"] = linear_color_interpolation(np.log10(z[z > 0]), val_min=np.log10(z_min),
                                                                 val_max=np.log10(z_max),
                                                                 col_min=c_min, col_max=c_max)
                    self.customized_cmap(values=np.geomspace(z_min, z_max, 255),
                                         colors=linear_color_interpolation(np.linspace(z_min, z_max, 255),
                                                                           col_min=c_min, col_max=c_max),
                                         scale=scale_z, **kwargs_colorbar)
                    self.compt_color += 2
                else:
                    c_min: str = l_colors[(self.compt_color + 1) % len(l_colors)]
                    c_med: str = l_colors[(self.compt_color + 2) % len(l_colors)]
                    c_max: str = l_colors[(self.compt_color + 3) % len(l_colors)]
                    ma: np.ndarray[bool] = z > 0.
                    kwargs["color"] = np.full(len(z), c_med)
                    z_min: np.float64 = np.min(z)
                    z_max: np.float64 = np.max(z)
                    if np.any(z > 0):
                        z_min_pos: np.float64 = np.min(z[z > 0])
                    else:
                        z_min_pos: np.float64 = 0.
                    if np.any(z < 0):
                        z_max_neg: np.float64 = np.max(z[z < 0])
                    else:
                        z_max_neg: np.float64 = 0.
                    if z_min_pos > 0.:
                        kwargs["color"][ma] = linear_color_interpolation(np.log10(z[ma]),
                                                                         val_min=np.log10(z_min_pos),
                                                                         val_max=np.log10(z_max),
                                                                         col_min=c_med, col_max=c_max)
                        z_colors1 = np.geomspace(z_min_pos, z_max, 255 // 2)
                        colors1 = linear_color_interpolation(np.linspace(np.log10(z_min_pos), np.log10(z_max),
                                                                         255 // 2),
                                                             col_min=c_med, col_max=c_max)
                    else:
                        z_colors1 = np.array([])
                        colors1 = np.array([])
                    if z_max_neg < 0.:
                        kwargs["color"][~ma] = linear_color_interpolation(np.log10(-z[~ma]),
                                                                          val_min=np.log10(-z_max_neg),
                                                                          val_max=np.log10(-z_min),
                                                                          col_min=c_med, col_max=c_min)
                        z_colors2 = -np.geomspace(-z_min, -z_max_neg, 255 // 2)
                        colors2 = linear_color_interpolation(np.linspace(np.log10(-z_min), np.log10(-z_max_neg),
                                                                         255 // 2),
                                                             col_min=c_med, col_max=c_min)
                    else:
                        z_colors2 = np.array([])
                        colors2 = np.array([])
                    z_colors: np.ndarray[np.float64] = np.append(z_colors2, z_colors1)
                    colors: np.ndarray[np.float64] = np.append(colors2, colors1)
                    self.customized_cmap(values=z_colors, colors=colors, scale=scale_z,
                                         **kwargs_colorbar)
            if "label" in kwargs and kwargs["label"] == "":
                del kwargs["label"]
            self.param_lines.append(kwargs)

        if hide and len(self.indexs_plot_lines) == 0:
            if dim_y == 1:
                self.indexs_plot_lines = list(np.arange(len(self.lines_x) - 1))
            elif dim_y > 1:
                self.indexs_plot_lines = list(np.arange(len(self.lines_x) - len(y)))
        elif not hide and len(self.indexs_plot_lines) != 0:
            if dim_y == 1:
                self.indexs_plot_lines.append(len(self.lines_x) - 1)
            elif dim_y > 1:
                self.indexs_plot_lines.extend(list(np.arange(len(self.lines_x) - len(y), len(self.lines_x))))

    def set_indexs_plot_lines(self, indexs=list | np.ndarray | str) -> None:
        """

        Set the order in which the lines are plotted. If the index size is smaller than the lists of x/y lines
        then some of them will not be plotted.

        Parameters
        ----------

        indexs : list | np.ndarray | str
            List lines' indexs to plot. The plotting order is the order of index.
            If index=="default", then all the lines will be plotted in the order in which they were saved

        Returns
        -------
            None

        """

        if isinstance(indexs, str) and indexs != "default":
            raise UserWarning("Graphique.set_order_plot_lines : The only string available as parameter is 'default',"
                              " not ", indexs)
        elif not isinstance(indexs, list | np.ndarray | str):
            raise UserWarning("Graphique.set_order_plot_lines : The only type available as parameter are "
                              "list, array or str not", type(indexs))
        elif (isinstance(indexs, list | np.ndarray) and (len(indexs) > len(self.lines_x)
                                                         or np.max(indexs) >= len(self.lines_x))):
            raise UserWarning("Graphique.set_order_plot_lines : there is at least one index in the index list",
                              np.max(indexs), "that is higher than the maximum possible index ", len(self.lines_x) - 1)
        self.indexs_plot_lines = list(indexs)

    def text(self, x: list | np.ndarray, y: list | np.ndarray,
             s: list | np.ndarray, axis_config: str = "bl", **kwargs) -> None:
        """
        Equivalent to ```plt.text```

        Parameters
        ----------

        x: list | np.ndarray
            Abscissa(s)
        y: list | np.ndarray
            Ordinate(s)
        s: list | np.ndarray
            Texts to plot
        axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
            The positions of x-y axis :
                - "bl" is x-axis on the bottom, y-axis on the left (default).
                - "tl" is x-axis on the top, y-axis on the left.
                - "br" is x-axis on the bottom, y-axis on the right.
                - "tr" is x-axis on the top, y-axis on the right.

        kwargs
            Additional argument to ```plot()``` function like linestyle, color....

        Returns
        -------
        None

        """

        if axis_config not in ["bl", "tl", "tr", "br"]:
            raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

        if isinstance(s, str) and isinstance(x, list | np.ndarray):
            s: np.ndarray = np.array([s for X in x])
        if (isinstance(x, list | np.ndarray) and isinstance(y, list | np.ndarray)
                and (len(np.shape(np.array(x))) == 1 and len(np.shape(np.array(y))) == 1
                     and len(np.array(x)) == len(np.array(y)))):
            args_auxi: list[dict] = [{"axis_config":axis_config} for X in x]
            for k in kwargs.keys():
                if isinstance(kwargs[k], list | np.ndarray) and len(kwargs[k]) == len(x):
                    for i in range(len(x)):
                        args_auxi[i][k] = kwargs[k][i]
                else:
                    for i in range(len(x)):
                        args_auxi[i][k] = kwargs[k]
            if "color" not in kwargs:
                for i in range(len(x)):
                    args_auxi[i]["color"] = l_colors[(len(self.lines_x) + i - 1) % len(l_colors)]
            for (X, Y, S, argsS) in zip(x, y, s, args_auxi):
                self.lines_t_x.append(X)
                self.lines_t_y.append(Y)
                self.lines_t_s.append(S)
                self.param_texts.append(argsS)
        elif (isinstance(x, list | np.ndarray) and isinstance(y, list | np.ndarray)
              and len(y) != len(x)):
            raise ValueError("Graphique.text : the ordinate list should have the same size than "
                             "the abscissa list len(x): "
                             + str(len(x)) + " y : " + str(np.shape(np.array(y))))
        elif (isinstance(x, list | np.ndarray) and isinstance(y, list | np.ndarray)
              and len(y) != len(s)):
            raise ValueError("Graphique.text : the ordinate list should have the same size than "
                             "the text list len(s): "
                             + str(len(s)) + " y : " + str(np.shape(np.array(y))))
        else:
            self.lines_t_x.append(x)
            self.lines_t_y.append(y)
            self.lines_t_s.append(s)
            if "color" not in kwargs:
                # args["color"] = 'C' + str(len(self.lines_x) % 10)
                kwargs["color"] = l_colors[(len(self.lines_x) - 1) % len(l_colors)]
            self.param_texts.append(kwargs)

    def point(self, xp: float | np.double, yp: float | np.double, marker: str = "o", hide: bool = False,
              axis_config: str = "bl", **kwargs) -> None:
        """
        Equivalent to Graphique.line([xp],[yp],**args)

        Parameters
        ----------
        xp: float | np.double
            Abscissa
        yp: float | np.double
             Ordinate
        marker: str, optional, default="o"
            The marker (ex ".", ",", "o", "v"...)
            see matplotlib documentation for all the possibility
        hide : bool, optional, default=False
            If True then the new line(s) is/are not plotted with the Graphique.
            To plot them, then change the plot order with self.set_indexs_plot_lines
        axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
            The positions of x-y axis :
                - "bl" is x-axis on the bottom, y-axis on the left (default).
                - "tl" is x-axis on the top, y-axis on the left.
                - "br" is x-axis on the bottom, y-axis on the right.
                - "tr" is x-axis on the top, y-axis on the right.

        kwargs
            Additional argument to plot() function like linestyle, color....

        Returns
        -------
            None

        """
        if axis_config not in ["bl", "tl", "tr", "br"]:
            raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

        self.line([xp], [yp], marker=marker, hide=hide, axis_config=axis_config,
                  **kwargs)

    def errorbar(
            self, x: list | np.ndarray, y: list | np.ndarray, err_y: list | np.ndarray,
            err_x: np.ndarray | list | None = None, marker: str = "", scale: str = "",
            hide: bool = False, axis_config: str = "bl", **kwargs) -> None:
        """
        Equivalent to plt.errorbar

        Parameters
        ----------
        x : list | array_like
            Abscissa
        y : list | array_like
            Ordinate
        err_y : list | array_like
            Error associated with y
        err_x : list | array_like
            Error associated with x
        marker : list[str] | array_like[str], str, optional, default=""
            The marker (ex ".", ",", "o", "v"...)
            see matplotlib documentation for all the possibility
        scale : str, optional, default="linear"
            The scales of (x, y) axis :
            - default : "" (linear scale for both x and y)
            - polar : polar projection : X=R and Y=Theta
            - loglog, logx, logy : Logarithmic scale for both, x or y axis
            - symloglog, symlogx, symlogy : Logarithmic scale for both, x or y axis with positive and négative values

        hide: bool, optional, default=False
            If True then the new line(s) is/are not plotted with the Graphique.
            To plot them, then change the plot order with self.set_indexs_plot_lines
        axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
            The positions of x-y axis :
                - "bl" is x-axis on the bottom, y-axis on the left (default).
                - "tl" is x-axis on the top, y-axis on the left.
                - "br" is x-axis on the bottom, y-axis on the right.
                - "tr" is x-axis on the top, y-axis on the right.

        kwargs
             Additional argument to plot() function like linestyle, color....

        Returns
        -------
        None

        """
        if axis_config not in ["bl", "tl", "tr", "br"]:
            raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

        if type(err_y) is str:
            raise TypeError("Graphique.errorbar : no error on the y axis are provided")
        if (len(err_y) != len(y) and len(err_y) != 2) or (len(err_y) == 2 and len(err_y[0]) != len(y)):
            raise ValueError(
                "Graphique.errorbar the size of ordinate's errors array should be equal to the size of "
                "ordinate array  : len(x)=" +
                str(len(x)) + " len(y)= " + str(len(y)) + " shape(err y)" + str(np.array(err_y).shape))
        elif (isinstance(y[0], list | np.ndarray) and isinstance(err_y[0], list | np.ndarray)
              and np.any([(len(Y) != len(errY) and len(errY) != 2)
                          or (len(errY) == 2 and len(Y) != len(errY[0]))
                          for (Y, errY) in zip(y, err_y)])):
            raise ValueError(
                "Graphique.errorbar the size of ordinate's errors array should be equal to the size of "
                "ordinate array")
        elif (isinstance(y[0], list | np.ndarray)
              and ((not isinstance(err_y[0], list | np.ndarray)) or len(err_y[0]) != 2)
              and np.any([(len(Y) != len(err_y) and len(err_y) != 2)
                          or (len(err_y) == 2 and len(Y) != len(err_y[0])) for Y in y])):
            raise ValueError(
                "Graphique.errorbar the size of ordinate's errors array should be equal to the size of "
                "ordinate array")
        if err_x is not None and (len(err_x) != len(x) or (len(err_x) == 2 and len(err_x[0]) != len(err_x))):
            raise ValueError(
                "Graphique.errorbar the size of abscissa's errors array should be equal to the size of "
                "abscissa array  : len(x)=" +
                str(len(x)) + " len(y)=" + str(len(y)) + " shape(err y)=" + str(np.array(err_y).shape))
        elif err_x is not None and (isinstance(y[0], list | np.ndarray) and isinstance(err_x[0], list | np.ndarray)
                                    and np.any([(len(X) != len(errX) and len(errX) != 2)
                                                or (len(errX) == 2 and len(errX[0]) != len(X))
                                                for (X, errX) in zip(x, err_x)])):
            raise ValueError(
                "Graphique.errorbar the size of abscissa's errors array should be equal to the size of"
                "ordinate array")
        elif err_x is not None and (isinstance(x[0], list | np.ndarray)
                                    and np.any([(len(X) != len(err_x) and len(err_x) != 2)
                                                or (len(err_x) == 2 and len(err_x[0]) != len(X))
                                                for X in x])):
            raise ValueError(
                "Graphique.errorbar the size of abscissa's errors array should be equal to the size of"
                "ordinate array")
        if scale == "":
            self.line(x, y, marker=marker, hide=hide, axis_config=axis_config, **kwargs)
        elif scale == "polaire":
            self.polar(x, y, marker=marker, hide=hide, axis_config=axis_config, **kwargs)
        elif scale == "loglog":
            self.loglog(x, y, marker=marker, hide=hide, axis_config=axis_config, **kwargs)
        elif scale == "logx":
            self.logx(x, y, marker=marker, hide=hide, axis_config=axis_config, **kwargs)
        elif scale == "logy":
            self.logy(x, y, marker=marker, hide=hide, axis_config=axis_config, **kwargs)
        elif scale == "symloglog":
            self.symloglog(x, y, marker=marker, hide=hide, axis_config=axis_config, **kwargs)
        elif scale == "symlogx":
            self.symlogx(x, y, marker=marker, hide=hide, axis_config=axis_config, **kwargs)
        elif scale == "symlogy":
            self.symlogy(x, y, marker=marker, hide=hide, axis_config=axis_config, **kwargs)
        else:
            raise (ValueError("The scale " + scale + """ is not available. Please use : "", "polar",
            "loglog", "logx", "logy", "symloglog", "symlogx", "symlogy" """))

        if isinstance(y[0], list | np.ndarray):
            for i in range(len(y) - 1, -1, -1):
                if isinstance(err_y[0], list | np.ndarray):
                    self.err_y[-1 - i] = err_y[-1 - i]
                else:
                    self.err_y[-1 - i] = err_y
                if err_x is not None and isinstance(err_x[0], list | np.ndarray):
                    self.err_x[-1 - i] = err_x[-1 - i]
                elif err_x is not None:
                    self.err_x[-1 - i] = err_x
                else:
                    self.err_x[-1 - i] = []
        else:
            self.err_y[-1] = err_y
            if err_x is not None:
                self.err_x[-1] = err_x
            else:
                self.err_x[-1] = []

    def errorplot(
            self, x: list | np.ndarray, y: list | np.ndarray, err_y: list | np.ndarray,
            marker: str = "", scale: str = "", hide: bool = False, axis_config: str = "bl", **kwargs) -> None:
        """
        Equivalent to plt.errorbar but the error is not represented by errorbars but by a
        uniform-colored polygon

        Parameters
        ----------
        x : list | array_like
            Abscissa
        y : list | array_like
            Ordinate
        err_y : list | array_like
            Error associated with y
        marker : list[str] | array_like[str], str, optional, default=""
            The marker (ex ".", ",", "o", "v"...)
            see matplotlib documentation for all the possibility
        scale : str, optional, default="linear"
            The scales of (x, y) axis :
            - default : "" (linear scale for both x and y)
            - polar : polar projection : X=R and Y=Theta
            - loglog, logx, logy : Logarithmic scale for both, x or y axis
            - symloglog, symlogx, symlogy : Logarithmic scale for both, x or y axis with positive and négative values

        hide: bool, optional, default=False
            If True then the new line(s) is/are not plotted with the Graphique.
            To plot them, then change the plot order with self.set_indexs_plot_lines
        kwargs
             Additional argument to plot() function like linestyle, color....
        axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
            The positions of x-y axis :
                - "bl" is x-axis on the bottom, y-axis on the left (default).
                - "tl" is x-axis on the top, y-axis on the left.
                - "br" is x-axis on the bottom, y-axis on the right.
                - "tr" is x-axis on the top, y-axis on the right.

        Returns
        -------
        None

        """
        if axis_config not in ["bl", "tl", "tr", "br"]:
            raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

        if type(err_y) is str:
            raise TypeError("Graphique.errorbar : no error on the y axis are provided")
        if len(err_y) != len(x):
            raise ValueError(
                "Graphique.errorbar the size of errors array should be equal to the size of"
                "abscissa array  : x :" +
                str(len(x)) + " y : " + str(len(y)) + " err y : " + str(len(err_y)))
        if scale == "":
            self.line(x, y, marker=marker, hide=hide, axis_config=axis_config, **kwargs)
        elif scale == "polaire":
            self.polar(x, y, marker=marker, hide=hide, axis_config=axis_config, **kwargs)
        elif scale == "loglog":
            self.loglog(x, y, marker=marker, hide=hide, axis_config=axis_config, **kwargs)
        elif scale == "logx":
            self.logx(x, y, marker=marker, hide=hide, axis_config=axis_config, **kwargs)
        elif scale == "logy":
            self.logy(x, y, marker=marker, hide=hide, axis_config=axis_config, **kwargs)
        elif scale == "symloglog":
            self.symloglog(x, y, marker=marker, hide=hide, axis_config=axis_config, **kwargs)
        elif scale == "symlogx":
            self.symlogx(x, y, marker=marker, hide=hide, axis_config=axis_config, **kwargs)
        elif scale == "symlogy":
            self.symlogy(x, y, marker=marker, hide=hide, axis_config=axis_config, **kwargs)
        else:
            raise (ValueError("The scale " + scale + """ is not available. Please use : "", "polar",
            "loglog", "logx", "logy", "symloglog", "symlogx", "symlogy" """))

        if isinstance(y, list | np.ndarray) and isinstance(err_y, list | np.ndarray):
            for (Y, errY) in zip(y, err_y):
                erry: list = list(Y + errY)
                erry2: list = list(Y - errY)
                erry2.reverse()
                erry.extend(erry2)
                x: list = list(x)
                x2: list = x.copy()
                x2.reverse()
                x.extend(x2)
                ind: np.ndarray = np.array([x, erry]).T
                self.polygon(ind, facecolor=self.param_lines[-1]["color"])
        elif isinstance(y, list | np.ndarray):
            for Y in y:
                erry: list = list(Y + err_y)
                erry2: list = list(Y - err_y)
                erry2.reverse()
                erry.extend(erry2)
                x: list = list(x)
                x2: list = x.copy()
                x2.reverse()
                x.extend(x2)
                ind: np.ndarray = np.array([x, erry]).T
                self.polygon(ind, facecolor=self.param_lines[-1]["color"])
        else:
            erry: list = list(y + err_y)
            erry2: list = list(y - err_y)
            erry2.reverse()
            erry.extend(erry2)
            x: list = list(x)
            x2: list = x.copy()
            x2.reverse()
            x.extend(x2)
            ind: np.ndarray = np.array([x, erry]).T
            self.polygon(ind, facecolor=self.param_lines[-1]["color"])

    def polar(self, r: list | np.ndarray, theta: list | np.ndarray,
              z: np.ndarray | list | None = None, marker: str = "",
              share_colorbar: bool = True, scale_z: str = "linear", hide: bool = False,
              kwargs_colorbar: dict | None = None, axis_config: str = "bl",
              **kwargs: dict) -> None:
        """

        Equivalent to self.line in polar projection

        Parameters
        ----------
        r : list | array_like
             Radius
        theta : list | array_like
            Angle(s)
        z : list | array_like, optional
            z-axis (represented by a colorscale)
        marker:  : list[str] | array_like, str, optional, default=""
            The marker (ex ".", ",", "o", "v"...)
            see matplotlib documentation for all the possibility
        share_colorbar : bool, optional, default=True
            If True(default) and z is not None, only one colorscale is used
            even if z is in two dimensions
        scale_z: str, optional, {"linear", "log", "symlog"}, default="linear"
            The scale of the z-axis (linear (default), log, symplog)
        hide : bool, optional, default=False
            If True then the new line(s) is/are not plotted with the Graphique.
            To plot them, then change the plot order with self.set_indexs_plot_lines
        kwargs_colorbar: dict
            Extra arguments for the colorbar (if z is not None)
        kwargs
            Additional argument to plot() function like linestyle, color...
        axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
            The positions of x-y axis :
                - "bl" is x-axis on the bottom, y-axis on the left (default).
                - "tl" is x-axis on the top, y-axis on the left.
                - "br" is x-axis on the bottom, y-axis on the right.
                - "tr" is x-axis on the top, y-axis on the right.

        Returns
        -------
        None

        Notes
        -------
        The order of first and second arguments is opposit to the matplotlib one :
        The first argument is the radius, then the angle

        """
        if axis_config not in ["bl", "tl", "tr", "br"]:
            raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

        self.line(r, theta, z=z, marker=marker, share_colorbar=share_colorbar,
                  scale_z=scale_z, hide=hide, kwargs_colorbar=kwargs_colorbar, axis_config=axis_config, **kwargs)
        self.config_ax(projection="polar")

    def loglog(self, x: np.ndarray | list, y: np.ndarray | list | None = None,
               z: np.ndarray | list | None = None,
               marker: str | list = "", share_colorbar: bool = True,
               scale_z: str = "linear", hide: bool = False, kwargs_colorbar: dict | None = None,
               axis_config: str = "bl", **kwargs) -> None:
        """
        Equivalent to self.line with a logarithmique scale for both x and y-axis:

        Parameters
        ----------
        x : array_like | list
            Abscissa(s)
        y : array_like | list, optional
            Ordinate(s), if None x became the ordinate and the abscissa is arange(len(x))
        z : array_like | list, optional
             z-axis (represented by a colorscale)
        marker : str | list[str] | array_like[str], optional, default=""
            The marker  ex ".", ",", "o", "v"... (see matplotlib documentation)
        share_colorbar : bool, optional, default=True
             If True (default) and z is not None, only one colorscale is used
             even if z is in two dimensions
        scale_z : str, {'linear', 'log', 'symlog'}, optional, default='linear'
            The scale of the z-axis
        hide : bool, optional, default=False
            If True then the new line(s) is/are not plotted with the Graphique.
            To plot them, then change the plot order with self.set_indexs_plot_lines
        kwargs_colorbar, optional
            Extra arguments for the colorbar (if z is not None)
        axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
            The positions of x-y axis :
                - "bl" is x-axis on the bottom, y-axis on the left (default).
                - "tl" is x-axis on the top, y-axis on the left.
                - "br" is x-axis on the bottom, y-axis on the right.
                - "tr" is x-axis on the top, y-axis on the right.

        kwargs, optional
            Additional argument to plot() function like linestyle, color....

        Returns
        -------
            None

        See Also
        --------
        Graphique.line : Build line(s) for the Graphique
        Graphique.logx : Graphique.line in log coordinate for x axis and linear for y axis
        Graphique.logy : Graphique.line in log coordinate for y axis and linear for x axis
        Graphique.point : To plot a single point
        Graphique.errorbar : To plot a line with errorbars
        Graphique.errorplot : To plot a line with errorbars represanted as filled area
        Graphique.polar : To plot a line in polar coordinates
        Graphique.symloglog : Similar to Graphique.loglog but boths negatives and positives values are represanted
        Graphique.symlogx : Similar to Graphique.logx but boths negatives and positives values are represanted
        Graphique.symlogy : Similar to Graphique.logy but boths negatives and positives values are represanted

        Notes
        -----
        This function has a small improuvment compared with plt.plot :

        if y is in two dimensions, the second dimension is plotted :
            - ```self.line(x,[y1,y2], *args)``` is equivalent to
                ```plt.plot(x, y1, *args)
                plt.plot(x, y2, *args)```
            - if y1 and y2 have not the same size:
                ```self.line([x1,x2],[y1, y2], *args)```
            - If others arguments are list of the same size of x and y, they are also split :
                ```self.line((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"]```
                is equivalent to
                ```plt.plot(x, y1, marker=".", label="Curve1")
                plt.plot(x, y2, marker=".", label="Curve2")```


        Examples
        --------
        >>> x = np.linspace(0, 10, 1000)
        >>> alpha = np.linspace(1, 5, 10)
        >>> colors = g.linear_color_interpolation(np.arange(len(alpha)), col_min=g.C1, col_max=g.C2)
        >>> gr = g.Graphique()
        >>> gr.loglog(x, [x*a for a in alpha], color=colors)
        >>> gr.customized_cmap(alpha, colors)
        >>> gr.show()

        """
        if axis_config not in ["bl", "tl", "tr", "br"]:
            raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

        self.line(x, y, z=z, marker=marker, share_colorbar=share_colorbar,
                  scale_z=scale_z, hide=hide, kwargs_colorbar=kwargs_colorbar, axis_config=axis_config, **kwargs)
        self.config_ax(xscale="log", yscale="log", axis=axis_config)

    def symloglog(self, x: np.ndarray | list, y: np.ndarray | list | None = None,
                  z: np.ndarray | list | None = None,
                  marker: str | list = "", share_colorbar: bool = True,
                  scale_z: str = "linear", hide: bool = False,
                  kwargs_colorbar: dict | None = None, axis_config: str = "bl", **kwargs) -> None:
        """
        Equivalent to self.line with a logarithmique scale for both x and y-axis
        Both the negative and positive parts of y are represanted:

        Parameters
        ----------
        x : array_like | list
            Abscissa(s)
        y : array_like | list, optional
            Ordinate(s), if None x became the ordinate and the abscissa is arange(len(x))
        z : array_like | list, optional
            z-axis (represented by a colorscale)
        marker : str | list[str] | array_like[str], optional, default=""
            The marker  ex ".", ",", "o", "v"... (see matplotlib documentation)
        share_colorbar : bool, optional, default=True
            If True (default) and z is not None, only one colorscale is used
            even if z is in two dimensions
        scale_z : str, {'linear', 'log', 'symlog'}, optional, default='linear'
            The scale of the z-axis
        hide : bool, optional, default=False
            If True then the new line(s) is/are not plotted with the Graphique.
            To plot them, then change the plot order with self.set_indexs_plot_lines
        kwargs_colorbar, optional
            Extra arguments for the colorbar (if z is not None)
        axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
            The positions of x-y axis :
                - "bl" is x-axis on the bottom, y-axis on the left (default).
                - "tl" is x-axis on the top, y-axis on the left.
                - "br" is x-axis on the bottom, y-axis on the right.
                - "tr" is x-axis on the top, y-axis on the right.

        kwargs, optional
            Additional argument to plot() function like linestyle, color....

        Returns
        -------
        None

        See Also
        --------
        Graphique.line : Build line(s) for the Graphique
        Graphique.loglog : Graphique.line in log coordinate for boths x and y axis
        Graphique.logx : Graphique.line in log coordinate for x axis and linear for y axis
        Graphique.logy : Graphique.line in log coordinate for y axis and linear for x axis
        Graphique.point : To plot a single point
        Graphique.errorbar : To plot a line with errorbars
        Graphique.errorplot : To plot a line with errorbars represanted as filled area
        Graphique.polar : To plot a line in polar coordinates
        Graphique.symloglog : Similar to Graphique.loglog but boths negatives and positives values are represanted
        Graphique.symlogx : Similar to Graphique.logx but boths negatives and positives values are represanted
        Graphique.symlogy : Similar to Graphique.logy but boths negatives and positives values are represanted

        Notes
        -------
        This function has a small improuvment compared with plt.plot :

        if y is in two dimensions, the second dimension is plotted :
            - ```self.symloglog(x,[y1,y2], *args)``` is equivalent to
                ```ax=plt.subplot()
                ax.plot(x, y1, *args)
                ax.plot(x, y2, *args)
                ax.set(xscale="symlog", yscale="symlog")```
            - if y1 and y2 have not the same size:
                ```self.line([x1,x2],[y1, y2], *args)```
            - If others arguments are list of the same size of x and y, they are also split :
                ```self.symploglog((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"])```
                is equivalent to
                ```ax=plt.subplot()
                ax.plot(x, y1, marker=".", label="Curve1")
                ax.plot(x, y2, marker=".", label="Curve2")
                ax.set(xscale="symlog", yscale="symlog")```


        Examples
        --------
        >>> x = np.linspace(-10, 10, 1000)
        >>> gr = g.Graphique()
        >>> gr.symloglog(x, np.tan(x))
        >>> gr.show()

        """
        if axis_config not in ["bl", "tl", "tr", "br"]:
            raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

        self.line(x, y, z=z, marker=marker, share_colorbar=share_colorbar,
                  scale_z=scale_z, hide=hide, kwargs_colorbar=kwargs_colorbar, axis_config=axis_config, **kwargs)
        self.config_ax(xscale="symlog", yscale="symlog", axis=axis_config)

    def logx(self, x: np.ndarray | list, y: np.ndarray | list | None = None,
             z: np.ndarray | list | None = None,
             marker: str | list = "", share_colorbar: bool = True,
             scale_z: str = "linear", hide: bool = False,
             kwargs_colorbar: dict | None = None, axis_config: str = "bl", **kwargs) -> None:
        """
        Equivalent to self.line with a logarithmique scale for x-axis:

        Parameters
        ----------
        x : array_like | list
            Abscissa(s)
        y : array_like | list, optional
            Ordinate(s), if None x became the ordinate and the abscissa is arange(len(x))
        z : array_like | list, optional
             z-axis (represented by a colorscale)
        marker : str | list[str] | array_like[str], optional, default=""
            The marker  ex ".", ",", "o", "v"... (see matplotlib documentation)
        share_colorbar : bool, optional, default=True
             If True (default) and z is not None, only one colorscale is used
             even if z is in two dimensions
        scale_z : str, {'linear', 'log', 'symlog'}, optional, default='linear'
            The scale of the z-axis
        hide : bool, optional, default=False
            If True then the new line(s) is/are not plotted with the Graphique.
            To plot them, then change the plot order with self.set_indexs_plot_lines
        kwargs_colorbar, optional
            Extra arguments for the colorbar (if z is not None)
        axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
            The positions of x-y axis :
                - "bl" is x-axis on the bottom, y-axis on the left (default).
                - "tl" is x-axis on the top, y-axis on the left.
                - "br" is x-axis on the bottom, y-axis on the right.
                - "tr" is x-axis on the top, y-axis on the right.

        kwargs, optional
            Additional argument to plot() function like linestyle, color....

        Returns
        -------
            None

        See Also
        --------
        Graphique.line : Build line(s) for the Graphique
        Graphique.loglog : Graphique.line in log coordinate for boths x and y axis
        Graphique.logy : Graphique.line in log coordinate for y axis and linear for x axis
        Graphique.point : To plot a single point
        Graphique.errorbar : To plot a line with errorbars
        Graphique.errorplot : To plot a line with errorbars represanted as filled area
        Graphique.polar : To plot a line in polar coordinates
        Graphique.symloglog : Similar to Graphique.loglog but boths negatives and positives values are represanted
        Graphique.symlogx : Similar to Graphique.logx but boths negatives and positives values are represanted
        Graphique.symlogy : Similar to Graphique.logy but boths negatives and positives values are represanted

        Notes
        -------
        This function has a small improuvment compared with plt.plot :

        if y is in two dimensions, the second dimension is plotted :

            - ```self.line(x,[y1,y2], *args)``` is equivalent to
                ```plt.plot(x, y1, *args)
                plt.plot(x, y2, *args)```
            - if y1 and y2 have not the same size:
                ```self.line([x1,x2],[y1, y2], *args)```
            - If others arguments are list of the same size of x and y, they are also split :
                ```self.line((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"])```
                is equivalent to
                ```plt.plot(x, y1, marker=".", label="Curve1")
                plt.plot(x, y2, marker=".", label="Curve2")```


        Examples
        --------
        >>> x = np.logspace(-10, 10, 1000)
        >>> gr = Graphique()
        >>> gr.logx(x, np.arctan(x))
        >>> gr.show()

        """
        if axis_config not in ["bl", "tl", "tr", "br"]:
            raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

        self.line(x, y, z=z, marker=marker, share_colorbar=share_colorbar,
                  scale_z=scale_z, hide=hide, kwargs_colorbar=kwargs_colorbar, axis_config=axis_config, **kwargs)
        self.config_ax(xscale="log", axis=axis_config)

    def symlogx(self, x: np.ndarray | list, y: np.ndarray | list | None = None,
                z: np.ndarray | list | None = None,
                marker: str | list = "", share_colorbar: bool = True,
                scale_z: str = "linear", hide: bool = False,
                kwargs_colorbar: dict | None = None, axis_config: str = "bl", **kwargs) -> None:
        """
        Equivalent to self.line with a logarithmique scale for both x-axis (both negative and positive
        part are represanted):

        Parameters
        ----------
        x : array_like | list
            Abscissa(s)
        y : array_like | list, optional
            Ordinate(s), if None x became the ordinate and the abscissa is arange(len(x))
        z : array_like | list, optional
             z-axis (represented by a colorscale)
        marker : str | list[str] | array_like[str], optional, default=""
            The marker  ex ".", ",", "o", "v"... (see matplotlib documentation)
        share_colorbar : bool, optional, default=True
             If True (default) and z is not None, only one colorscale is used
             even if z is in two dimensions
        scale_z : str, {'linear', 'log', 'symlog'}, optional, default='linear'
            The scale of the z-axis
        hide : bool, optional, default=False
            If True then the new line(s) is/are not plotted with the Graphique.
            To plot them, then change the plot order with self.set_indexs_plot_lines
        kwargs_colorbar, optional
            Extra arguments for the colorbar (if z is not None)
        axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
            The positions of x-y axis :
                - "bl" is x-axis on the bottom, y-axis on the left (default).
                - "tl" is x-axis on the top, y-axis on the left.
                - "br" is x-axis on the bottom, y-axis on the right.
                - "tr" is x-axis on the top, y-axis on the right.

        kwargs, optional
            Additional argument to plot() function like linestyle, color....

        Returns
        -------
            None

        See Also
        --------
        Graphique.line : Build line(s) for the Graphique
        Graphique.loglog : Graphique.line in log coordinate for boths x and y axis
        Graphique.logx : Graphique.line in log coordinate for x axis and linear for y axis
        Graphique.logy : Graphique.line in log coordinate for y axis and linear for x axis
        Graphique.point : To plot a single point
        Graphique.errorbar : To plot a line with errorbars
        Graphique.errorplot : To plot a line with errorbars represanted as filled area
        Graphique.polar : To plot a line in polar coordinates
        Graphique.symloglog : Similar to Graphique.loglog but boths negatives and positives values are represanted
        Graphique.symlogy : Similar to Graphique.logy but boths negatives and positives values are represanted

        Notes
        -------
        This function has a small improuvment compared with plt.plot :

        if y is in two dimensions, the second dimension is plotted :
            - ```self.line(x,[y1,y2], *args)``` is equivalent to
                ```plt.plot(x, y1, *args)
                plt.plot(x, y2, *args)```
            - if y1 and y2 have not the same size:
                ```self.line([x1,x2],[y1, y2], *args)```
            - If others arguments are list of the same size of x and y, they are also split :
                ```self.line((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"])```
                is equivalent to
                ```plt.plot(x, y1, marker=".", label="Curve1")
                plt.plot(x, y2, marker=".", label="Curve2")```


        Examples
        --------
        >>> x = np.append(-np.logspace(10, -10, 1000), np.logspace(-10, 10, 1000))
        >>> gr = Graphique()
        >>> gr.symlogx(x, np.arctan(x))
        >>> gr.show()

        """
        if axis_config not in ["bl", "tl", "tr", "br"]:
            raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

        self.line(x, y, z=z, marker=marker, share_colorbar=share_colorbar,
                  scale_z=scale_z, hide=hide, kwargs_colorbar=kwargs_colorbar, axis_config=axis_config, **kwargs)
        self.config_ax(xscale="symlog", axis=axis_config)

    def logy(self, x: np.ndarray | list, y: np.ndarray | list | None = None,
             z: np.ndarray | list | None = None,
             marker: str | list = "", share_colorbar: bool = True,
             scale_z: str = "linear", hide: bool = False,
             kwargs_colorbar: dict | None = None, axis_config: str = "bl", **kwargs) -> None:
        """
        Equivalent to ```self.line``` with a logarithmique scale for y-axis:

        Parameters
        ----------
        x : array_like | list
            Abscissa(s)
        y : array_like | list, optional
            Ordinate(s), if None x became the ordinate and the abscissa is arange(len(x))
        z : array_like | list, optional
             z-axis (represented by a colorscale)
        marker : str | list[str] | array_like[str], optional, default=""
            The marker  ex ".", ",", "o", "v"... (see matplotlib documentation)
        share_colorbar : bool, optional, default=True
             If True (default) and z is not None, only one colorscale is used
             even if z is in two dimensions
        scale_z : str, {'linear', 'log', 'symlog'}, optional, default='linear'
            The scale of the z-axis
        hide : bool, optional, default=False
            If True then the new line(s) is/are not plotted with the Graphique.
            To plot them, then change the plot order with self.set_indexs_plot_lines
        kwargs_colorbar, optional
            Extra arguments for the colorbar (if z is not None)
        axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
            The positions of x-y axis :
                - "bl" is x-axis on the bottom, y-axis on the left (default).
                - "tl" is x-axis on the top, y-axis on the left.
                - "br" is x-axis on the bottom, y-axis on the right.
                - "tr" is x-axis on the top, y-axis on the right.

        kwargs, optional
            Additional argument to plot() function like linestyle, color....

        Returns
        -------
            None

        See Also
        --------
        Graphique.line : Build line(s) for the Graphique
        Graphique.loglog : Graphique.line in log coordinate for boths x and y axis
        Graphique.logx : Graphique.line in log coordinate for x axis and linear for y axis
        Graphique.point : To plot a single point
        Graphique.errorbar : To plot a line with errorbars
        Graphique.errorplot : To plot a line with errorbars represanted as filled area
        Graphique.polar : To plot a line in polar coordinates
        Graphique.symloglog : Similar to Graphique.loglog but boths negatives and positives values are represanted
        Graphique.symlogx : Similar to Graphique.logx but boths negatives and positives values are represanted
        Graphique.symlogy : Similar to Graphique.logy but boths negatives and positives values are represanted

        Notes
        -------
        This function has a small improuvment compared with plt.plot :

        if y is in two dimensions, the second dimension is plotted :

            - ```self.line(x,[y1,y2], *args)``` is equivalent to
                ```plt.plot(x, y1, *args)
                plt.plot(x, y2, *args)```
            - if y1 and y2 have not the same size:
                ```self.line([x1,x2],[y1, y2], *args)```
            - If others arguments are list of the same size of x and y, they are also split :
                ```self.line((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"])```
                is equivalent to
                ```plt.plot(x, y1, marker=".", label="Curve1")
                plt.plot(x, y2, marker=".", label="Curve2")```


        Examples
        --------
        >>> x = np.logspace(-10, 10, 1000)
        >>> gr = Graphique()
        >>> gr.logy(x, np.arctan(x))
        >>> gr.show()

        """
        if axis_config not in ["bl", "tl", "tr", "br"]:
            raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

        self.line(x, y, z=z, marker=marker, share_colorbar=share_colorbar,
                  scale_z=scale_z, hide=hide, kwargs_colorbar=kwargs_colorbar, axis_config=axis_config, **kwargs)
        self.config_ax(yscale="log", axis=axis_config)

    def symlogy(self, x: np.ndarray | list, y: np.ndarray | list | None = None,
                z: np.ndarray | list | None = None,
                marker: str | list = "", share_colorbar: bool = True,
                scale_z: str = "linear", hide: bool = False,
                kwargs_colorbar: dict | None = None, axis_config: str = "bl", **kwargs) -> None:
        """
        Equivalent to ```self.line``` with a logarithmique scale for y-axis (both positive and negative
        part are represanted):

        Parameters
        ----------
        x : array_like | list
            Abscissa(s)
        y : array_like | list, optional
            Ordinate(s), if None x became the ordinate and the abscissa is arange(len(x))
        z : array_like | list, optional
             z-axis (represented by a colorscale)
        marker : str | list[str] | array_like[str], optional, default=""
            The marker  ex ".", ",", "o", "v"... (see matplotlib documentation)
        share_colorbar : bool, optional, default=True
             If True (default) and z is not None, only one colorscale is used
             even if z is in two dimensions
        scale_z : str, {'linear', 'log', 'symlog'}, optional, default='linear'
            The scale of the z-axis
        hide : bool, optional, default=False
            If True then the new line(s) is/are not plotted with the Graphique.
            To plot them, then change the plot order with self.set_indexs_plot_lines
        kwargs_colorbar, optional
            Extra arguments for the colorbar (if z is not None)
        axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
            The positions of x-y axis :
                - "bl" is x-axis on the bottom, y-axis on the left (default).
                - "tl" is x-axis on the top, y-axis on the left.
                - "br" is x-axis on the bottom, y-axis on the right.
                - "tr" is x-axis on the top, y-axis on the right.

        kwargs, optional
            Additional argument to plot() function like linestyle, color....

        Returns
        -------
            None

        See Also
        --------
        Graphique.line : Build line(s) for the Graphique
        Graphique.loglog : Graphique.line in log coordinate for boths x and y axis
        Graphique.logy : Graphique.line in log coordinate for y axis and linear for x axis
        Graphique.point : To plot a single point
        Graphique.errorbar : To plot a line with errorbars
        Graphique.errorplot : To plot a line with errorbars represanted as filled area
        Graphique.polar : To plot a line in polar coordinates
        Graphique.symloglog : Similar to Graphique.loglog but boths negatives and positives values are represanted
        Graphique.symlogx : Similar to Graphique.logx but boths negatives and positives values are represanted

        Notes
        -------
        This function has a small improuvment compared with plt.plot :

        if y is in two dimensions, the second dimension is plotted :
            - ```self.symlogy(x,[y1,y2], *args)``` is equivalent to
                ```ax = plt.subplot()
                ax.plot(x, y1, *args)
                ax.plot(x, y2, *args)
                ax.set(yscale="symplog")```
            - if y1 and y2 have not the same size:
                ```self.symlogy([x1,x2],[y1, y2], *args)```
            - If others arguments are list of the same size of x and y, they are also split :
                ```self.symplogy((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"])```
                is equivalent to
                ```ax=plt.subplot()
                ax.plot(x, y1, marker=".", label="Curve1")
                ax.plot(x, y2, marker=".", label="Curve2")
                ax.set(yscale="symlog")```


        Examples
        --------
        >>> x = np.linspace(0,np.pi,1000)
        >>> gr = g.Graphique()
        >>> gr.symlogy(x, np.tan(x))
        >>> gr.show()

        """
        if axis_config not in ["bl", "tl", "tr", "br"]:
            raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

        self.line(x, y, z=z, marker=marker, share_colorbar=share_colorbar,
                  scale_z=scale_z, hide=hide, kwargs_colorbar=kwargs_colorbar, **kwargs)
        self.config_ax(yscale="symlog", axis=axis_config)

    def histogram(
            self, values: np.ndarray, weights: np.ndarray | None = None,
            normalization: bool = True, statistic: str = 'sum', bins: int = 10, stat_args: dict = None,
            **kwargs) -> None:
        """
        Plot the histogram of values

        Parameters
        ----------
        values : array_like
            The values to histogramed
        weights : array_like, optional
            The weights to be applied to values
        normalization : bool, optional, default=True
            If the histogram is normalized or not
        statistic : str, optional, default="sum"
            The statistic to compute.
            The following statistics are available

                - 'mean': compute the mean of values for points within each bin. Empty bins will be represented by NaN.
                - 'std': compute the standard deviation within each bin. This is implicitly calculated with ddof=0.
                - 'median': compute the median of values for points within each bin. Empty bins will be represented by NaN.
                - 'count': compute the count of points within each bin. This is identical to an unweighted histogram. values array is not
                    referenced.
                - 'sum': compute the sum of values for points within each bin. This is identical to a weighted histogram.
                - 'min': compute the minimum of values for points within each bin. Empty bins will be represented by NaN.
                - 'max': compute the maximum of values for point within each bin. Empty bins will be represented by NaN.

        bins : int, optional, default=10
            Number of bins in the histogram
        stat_args, dict, optional
            Additionals argument for `sp.binned_statistic`
        kwargs
             Additionals argument for plt.bars

        Returns
        -------
        None

        See also
        --------
        scipy.stats.binned_statistic

        """
        if weights is None:
            weights = np.ones(len(values))
        if stat_args is None:
            stat_args = dict()

        vals, bds, indices = sp.binned_statistic(
            values, weights, statistic, bins, **stat_args)

        if normalization:
            vals /= len(values)
            vals /= bds[1:] - bds[:-1]

        if "color" not in kwargs:
            kwargs["color"] = l_colors[(self.compt_color + 1) % len(l_colors)]
            self.compt_color += 1

        self.bords_histogramme.append(bds)
        self.vals_histogramme.append(vals)
        self.param_histogrammes.append(kwargs)

    def image(self, array_image: np.ndarray,
              x_axe: list | np.ndarray | None = None, y_axe: list | np.ndarray | None = None,
              colorscale: str = "linear",
              cmap: str = "default", colorbar_ticks: list | np.ndarray | None = None,
              colorbar_label: str = "", kwargs_colorbar: dict | None = None,
              colorbar_index: int = ii_max,
              vmin: np.float64 = -np.inf, vmax: np.float64 = np.inf, axis_config: str = "bl",
              **kwargs) -> None:
        """

        Plot the array image through plt.pcolor or plt.imshow for 3 color images

        Parameters
        ----------
        array_image : np.ndarray
            The matrix (2D, or 3D for colored images (the color is on the third axis)) to be plotted
        x_axe : list | np.ndarray, optional, default=np.linespace(0,array_image.shape[0])
            The x-axes coordinate (for the array), only for 2d array_image
        y_axe : list | np.ndarray, optional, default=np.linespace(0,array_image.shape[1])
            The y-axes coordinate (for the array), only for 2d array_image
        colorscale : str, optional, default="linear", {"linear", "log", "symlog"}
            The scale for the colorbar
        cmap : str, optional, default="default"
            The colormap, default a linear color interpolation between two colors CX
        colorbar_ticks : list | array_like, optional
            The colorbar's ticks
        colorbar_label : str, optional, default=""
            The colorbar's label
        kwargs_colorbar : dict, optional
            Additional arguments for the colorbar :

                - location: str, {'right', 'top', 'bottom', 'left'}
                    Indicate where the colorbar should be plotted
                - scale: str, {'linear', 'log', 'symlog'}
                    The scale of the colorbar
                - ticks: list | array_like
                - format: str
                    ticks' format
                - label: str
                    The label to plot along the colorbar
                - size: float, default=0.01
                    relative width of the colorbar
                - fraction: float, default=1
                    relative hight of the colorbar
                - space_between: float, default=0.01
                    relative space between colorsbars (and the plot)

        colorbar_index : int, optional
            The index of a previouly defined colorbar, to use insted of building a new one, all the others colorbar are
            ignored if provided
        vmin : np.float64, optional, default : the minimum of array_image
            The minimum value for the colorbar
        vmax : np.float64, optional, default : the maximum of array_image
            The maximum value for the colorbar
        axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
            The positions of x-y axis :
                - "bl" is x-axis on the bottom, y-axis on the left (default).
                - "tl" is x-axis on the top, y-axis on the left.
                - "br" is x-axis on the bottom, y-axis on the right.
                - "tr" is x-axis on the top, y-axis on the right.

        kwargs
            Additionals arguments for pcolor

        Returns
        -------
            None

        See Also
        --------

        matplotlib.pyplot.pcolor
            Use to plot 2d array images
        matplotlib.pyplot.imshow
            Use to plot 3-colors images
        Graphique.contours
            To draw levels lines onto the image

        """
        if axis_config not in ["bl", "tl", "tr", "br"]:
            raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)
        kwargs["axis_config"] = axis_config
        if vmin == -np.inf:
            vmin: np.float64 = np.nanmin(array_image)
        else:
            array_image[array_image < vmin] = vmin
        if vmax == np.inf:
            vmax: np.float64 = np.nanmax(array_image)
        else:
            array_image[array_image > vmax] = vmax
        if kwargs_colorbar is None:
            kwargs_colorbar = {}
        if colorscale == "linear":
            color_values: np.ndarray[np.float64] = np.linspace(vmin, vmax, 255)
        elif colorscale == "log":
            if vmin <= 0. and np.any(array_image > 0.):
                vmin: np.float64 = np.nanmin(array_image[array_image > 0.])
            else:
                raise UserWarning("Graphique.image : There is no positive values to plot with the log scale")
            color_values: np.ndarray[np.float64] = np.geomspace(vmin, vmax, 255)
        elif colorscale == "symlog":
            if vmin > 0:
                color_values: np.ndarray[np.float64] = np.geomspace(vmin, vmax, 255)
            elif vmax < 0:
                color_values: np.ndarray[np.float64] = np.geomspace(-vmin, -vmax, 255)
            else:
                vneg_sup: np.float64 = np.nanmax(array_image[array_image < 0.])
                vpos_inf: np.float64 = np.nanmin(array_image[array_image > 0.])
                color_values: np.ndarray[np.float64] = np.append(np.geomspace(-vmin, -vneg_sup,
                                                                              int(abs(np.log10(vmin / vneg_sup)
                                                                                      / (np.log10(vmin / vneg_sup)
                                                                                         + np.log10(vmax / vpos_inf)))
                                                                                  * 255)),
                                                                 np.geomspace(vpos_inf, vmax,
                                                                              int(abs(np.log10(vmax / vpos_inf)
                                                                                      / (np.log10(vmin / vneg_sup)
                                                                                         + np.log10(vmax / vpos_inf)))
                                                                                  * 255)))
        else:
            color_values: np.ndarray[np.float64] = colorscale

        if len(array_image.shape) == 3:
            if array_image.shape[2] > 4:
                raise UserWarning("Graphique.image : The array_image have not the right shape to plot a color image")
            array_image = np.interp(array_image, color_values, np.linspace(0, 1, len(color_values)))
            if cmap != "default":
                print("Warning Graphique.image : for colored images, the colorbars parameters are ignored")
            if "location" not in kwargs_colorbar.keys():
                kwargs_colorbar["location"] = 'right'
            if "size" not in kwargs_colorbar.keys():
                kwargs_colorbar["size"] = 0.02

            colors_r: np.ndarray[str] = linear_color_interpolation(np.arange(len(color_values)),
                                                                   col_min=to_hex((0, 0, 0)),
                                                                   col_max=to_hex((1, 0, 0)))

            self.customized_cmap(color_values, colors_r, ticks=colorbar_ticks, label=colorbar_label,
                                 **kwargs_colorbar)
            self.index_colorbar_image.append(len(self.custum_colorbar_colors) - 1)

            colors_v: np.ndarray[str] = linear_color_interpolation(np.arange(len(color_values)),
                                                                   col_min=to_hex((0, 0, 0)),
                                                                   col_max=to_hex((0, 1, 0)))
            self.customized_cmap(color_values, colors_v, ticks="", label="", share_axis=True,
                                 space_between=0.005, **kwargs_colorbar)
            self.index_colorbar_image.append(len(self.custum_colorbar_colors) - 1)

            colors_b: np.ndarray[str] = linear_color_interpolation(np.arange(len(color_values)),
                                                                   col_min=to_hex((0, 0, 0)),
                                                                   col_max=to_hex((0, 0, 1)))
            self.customized_cmap(color_values, colors_b, ticks="", label="", share_axis=True,
                                 space_between=0.005, **kwargs_colorbar)
            self.index_colorbar_image.append(len(self.custum_colorbar_colors) - 1)
        elif colorbar_index == ii_max:
            if cmap == "default":
                colors: np.ndarray[str] = linear_color_interpolation(np.arange(len(color_values)),
                                                                     col_min=l_colors[(self.compt_color + 1)
                                                                                      % len(l_colors)],
                                                                     col_max=l_colors[(self.compt_color + 2)
                                                                                      % len(l_colors)])
                self.compt_color += 2
            elif cmap not in list(plt.colormaps):
                raise UserWarning("Graphique.image : the colorbar ", cmap,
                                  "isn't awalible. Please use ", list(plt.colormaps))
            else:
                cmap_auxi = plt.get_cmap(cmap)
                colors: np.ndarray[str] = np.array([to_hex(col) for col in
                                                    cmap_auxi(np.linspace(0, 1, len(color_values)))])
            if kwargs_colorbar is None:
                kwargs_colorbar = dict()
            self.customized_cmap(color_values, colors, ticks=colorbar_ticks, label=colorbar_label, **kwargs_colorbar)
            self.index_colorbar_image.append(len(self.custum_colorbar_colors) - 1)
        else:
            self.index_colorbar_image.append(int(cmap))
        self.array_image = array_image
        self.x_axe_image = np.array(x_axe)
        self.y_axe_image = np.array(y_axe)
        self.param_image = kwargs

    def contours(
            self, levels: int | np.ndarray | list | None = None, array_contours: np.ndarray | None = None,
            x_axe: list | np.ndarray | None = None,
            y_axe: list | np.ndarray | None = None, labels: list | np.ndarray | None = None,
            labels_mask: np.ndarray | None = None, axis_config: str = "bl", **kwargs):
        """

        Plot the level lines associated to self.array_image or array_contours

        Parameters
        ----------
        levels : int | array_like | list, optional
            Number (or list of) levels to plot
        array_contours : array_like, optional, default=self.arry_image
            If not None, the reference array to determine the level
        x_axe : array_like | list, optional
            the x-axes coordinate (for the array if array_contour is not None)
        y_axe : array_like | list, optional
            the y-axes coordinate (for the array if array_contour is not None)
        labels : array_like | list, optional
            the labels of each level line
        labels_mask : array_like | list, optional
            the mask of levels line to show the labels
        axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
            The positions of x-y axis :
                - "bl" is x-axis on the bottom, y-axis on the left (default).
                - "tl" is x-axis on the top, y-axis on the left.
                - "br" is x-axis on the bottom, y-axis on the right.
                - "tr" is x-axis on the top, y-axis on the right.

        kwargs
            additional arguments

        Returns
        -------
        None


        See Also
        --------
        matplotlib.pyplot.contour
            Used to plot the levels lines
        Graphique.image
            To plot an image, this image can be used as a reference for the levels lines

        """
        if axis_config not in ["bl", "tl", "tr", "br"]:
            raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)
        kwargs["axis_config"] = axis_config
        idx_levels: np.ndarray | None = None
        if type(levels) is list or type(levels) is np.ndarray:
            idx_levels = np.argsort(levels)
            levels = levels[idx_levels]

        if "colors" in kwargs.keys() and (type(kwargs["colors"]) is list
                                          or type(kwargs["colors"]) is np.ndarray):
            self.color_label_contours = kwargs["colors"]
            del kwargs["colors"]

        if levels is not None:
            kwargs['levels'] = levels
            if labels is not None:
                if len(labels) != len(levels):
                    raise UserWarning("Graphique.contours : the labels size should be equal to the levels size: levels",
                                      len(levels), "labels :", len(labels))
                self.clabels = labels[idx_levels]

                if labels_mask is not None:
                    if len(labels_mask) != len(levels):
                        raise UserWarning("Graphique.contours : the labels_mask size should be equal "
                                          "to the levels/labels size: levels",
                                          len(levels), "labels_mask :", len(labels_mask))
                    self.clabels_mask = labels_mask[idx_levels]
        if array_contours is None:
            self.tab_contours_is_image = True
            if type(kwargs['levels']) is int:
                self.nb_contours = kwargs['levels']
                del kwargs['levels']
            else:
                self.nb_contours = len(kwargs['levels'])
                self.levels = kwargs['levels']
                del kwargs['levels']
            liste_intersect: list[str] = ['alpha0', 'vmin', 'vmax', 'norm']
            if "colors" not in kwargs:
                liste_intersect.append("cmap")
            for p in liste_intersect:
                if p in self.param_image:
                    self.param_contours[p] = self.param_image[p]
            self.param_contours.update(kwargs)
        else:
            self.array_contours = array_contours
            self.tab_contours_is_image = False
            self.x_axe_contours = x_axe
            self.y_axe_contours = y_axe
            self.param_contours = kwargs

    def polygon(self, ind, alpha: float | np.double = 0.7, facecolor: str = 'C3', plot_borders: bool = True,
                axis_config: str = "bl", **kwargs) -> None:
        """

        Plot a uniformly colored polygon

        Parameters
        ----------
        ind
            2-dimensional array/list of the coordinate of the polygon characteristics points
            ind[:, 0] point's abscissas
            ind[:, 1] point's ordinate
        alpha
            transparency (between 0 and 1, default 0.7)
        facecolor
            Polygon's color
        plot_borders
            If True (default) plot a line at the polygon's border
        axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
            The positions of x-y axis :
                - "bl" is x-axis on the bottom, y-axis on the left (default).
                - "tl" is x-axis on the top, y-axis on the left.
                - "br" is x-axis on the bottom, y-axis on the right.
                - "tr" is x-axis on the top, y-axis on the right.

        kwargs
            Extras arguments for matplotlib.patches.PathPatch

        Returns
        -------
        None

        See Also
        --------
        matplotlib.patches.PathPatch
            Use to draw the polygon

        """
        if axis_config not in ["bl", "tl", "tr", "br"]:
            raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)
        kwargs["axis_config"] = axis_config
        self.index_polygons.append(ind)
        kwargs["alpha"] = alpha
        kwargs['facecolor'] = facecolor
        if plot_borders and "edgecolor" not in kwargs.keys():
            kwargs['edgecolor'] = facecolor
        self.param_polygons.append(kwargs)

    def config_ax(self, axis: str = "bl", **kwargs) -> None:
        """
        Additionals configurations for ax

        Parameters
        ----------
        axis : str, optional, {"bl", "tl", "br", "tr", "all"}, default="bl"
            The positions of x-y axis to config:
                - "bl" is x-axis on the bottom, y-axis on the left (default).
                - "tl" is x-axis on the top, y-axis on the left.
                - "br" is x-axis on the bottom, y-axis on the right.
                - "tr" is x-axis on the top, y-axis on the right.
                - "all" will configurate aull axis

        kwargs
            Keywords available (see matplotlib documentation):
            - ```sharex```, ```sharey``` Axes, optional
            The x- or y-axis isshared with the x- or y-axis in the input Axes.
            Note that it is not possible to unshare axes.
            - ```frameonbool``` : default=True
            Whether the Axes frame is visible.
            - ```box_aspect``` : float, optional
            Set a fixed aspect for the Axes box,
            i.e. the ratio of height to width. See set_box_aspect for details.
            - ```forward_navigation_events``` : bool or "auto", default: "auto"
            Control whether pan/zoom events are passed through to Axes below this one. "auto" is True for axes with an
            invisible patch and False otherwise.
            - Other optional keyword arguments:
            -- adjustable {'box', 'datalim'}
            -- agg_filter : a filter function, which takes a (m, n, 3) float array and a dpi value, and returns
            a (m, n, 3) array and two offsets from the bottom left corner of the image
            -- alpha : scalar or None
            -- anchor : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', ...}
            -- animated : bool
            -- aspect : {'auto', 'equal'} or float
            -- autoscale_on : bool
            -- autoscalex_on
            -- autoscaley_on
            -- axes_locator : Callable[[Axes, Renderer], Bbox]
            -- axisbelow : bool or 'line'
            -- box_aspect : float or None
            -- clip_on : bool
            -- facecolor or fc : color
            -- figure : Figure
            -- forward_navigation_events : bool or "auto"
            -- frame_on : bool
            -- gid : str
            -- in_layout : bool
            -- label : object
            -- mouseover : bool
            -- navigate : bool
            -- navigate_mode
            -- picker : None or bool or float
            -- position : [left, bottom, width, height]
            -- rasterization_zorder : float or None
            -- rasterized : bool
            -- sketch_params : (scale: float, length: float, randomness: float)
            -- snap : bool or None
            -- subplotspec
            -- title : str
            -- url : str
            -- visible : bool
            -- xbound : (lower: float, upper: float)
            -- xlabel : str
            -- xlim : (left: float, right: float)
            -- xmargin : float greater than -0.5
            -- xscale
            -- xticklabels
            -- xticks
            -- ybound : (lower: float, upper: float)
            -- ylabel : str
            -- ylim : (bottom: float, top: float)
            -- ymargin : float greater than -0.5
            -- yscale
            -- yticklabels
            -- yticks
            -- zorder : float

        Returns
        -------
        None

        See Also
        --------

        matplotlib.axes.set
            The function used with kwargs

        """
        if 'xticks' in kwargs:
            if axis == "bl" or axis == "all":
                self.x_axe[0] = kwargs['xticks']
            if axis == "tl" or axis == "all":
                self.x_axe[1] = kwargs['xticks']
            if axis == "tr" or axis == "all":
                self.x_axe[2] = kwargs['xticks']
            if axis == "br" or axis == "all":
                self.x_axe[3] = kwargs['xticks']
            del kwargs['xticks']
        if 'yticks' in kwargs:
            if axis == "bl" or axis == "all":
                self.y_axe[0] = kwargs['yticks']
            if axis == "tl" or axis == "all":
                self.y_axe[1] = kwargs['yticks']
            if axis == "tr" or axis == "all":
                self.y_axe[2] = kwargs['yticks']
            if axis == "br" or axis == "all":
                self.y_axe[3] = kwargs['yticks']
            del kwargs['yticks']
        if 'xticklabels' in kwargs:
            if axis == "bl" or axis == "all":
                self.labels_x_ticks[0] = kwargs['xticklabels']
            if axis == "tl" or axis == "all":
                self.labels_x_ticks[1] = kwargs['xticklabels']
            if axis == "tr" or axis == "all":
                self.labels_x_ticks[2] = kwargs['xticklabels']
            if axis == "br" or axis == "all":
                self.labels_x_ticks[3] = kwargs['xticklabels']
            del kwargs['xticklabels']
        if "yticklabels" in kwargs:
            if axis == "bl" or axis == "all":
                self.labels_y_ticks[0] = kwargs['yticklabels']
            if axis == "tl" or axis == "all":
                self.labels_y_ticks[1] = kwargs['yticklabels']
            if axis == "tr" or axis == "all":
                self.labels_y_ticks[2] = kwargs['yticklabels']
            if axis == "br" or axis == "all":
                self.labels_y_ticks[3] = kwargs['yticklabels']
            del kwargs['yticklabels']
        if "Figure" in kwargs:
            self.fig = kwargs["Figure"]
            del kwargs["Figure"]

        if (axis == "bl") | (axis == "all"):
            self.param_ax.update(kwargs)
        if (axis == "tl") | (axis == "all"):
            self.param_ax_tl.update(kwargs)
        if (axis == "tr") | (axis == "all"):
            self.param_ax_tr.update(kwargs)
        if (axis == "br") | (axis == "all"):
            self.param_ax_br.update(kwargs)

    def config_legende(self, **kwargs) -> None:
        """
        To set additionals parameters for the legend plotting

        Parameters
        ----------
        kwargs
            additionals parameters for the legend
            (see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html)

            - locstr  default: rcParams["legend.loc"] (default: 'best')  The location of the legend.
                The strings 'upper left', 'upper right', 'lower left', 'lower right' place the legend at
                the corresponding corner of the axes.
                The strings 'upper center', 'lower center', 'center left', 'center right' place the
                legend at the center of the corresponding edge of the axes.
                The string 'center' places the legend at the center of the axes.
                The string 'best' places the legend at the location, among the nine locations defined so far, with the
                minimum overlap with other drawn artists. This option can be quite slow for plots with large amounts
                of data; your plotting speed may benefit from providing a specific location.
                The location can also be a 2-tuple giving the coordinates of the lower-left corner of the legend in
                axes coordinates (in which case bbox_to_anchor will be ignored).
                For back-compatibility, 'center right' (but no other location) can also be spelled 'right', and each
                "string" location can also be given as a numeric value:

            - bbox_to_anchorBboxBase, 2-tuple, or 4-tuple of floats
                Box that is used to position the legend in conjunction with loc.
                Defaults to axes.bbox (if called as a method to Axes.legend) or figure.bbox (if Figure.legend).
                This argument allows arbitrary placement of the legend.
                Bbox coordinates are interpreted in the coordinate system given by bbox_transform, with the default
                transform Axes or Figure coordinates, depending on which legend is called.
                If a 4-tuple or BboxBase is given, then it specifies the bbox (x, y, width, height) that the legend is
                placed in. To put the legend in the best location in the bottom right quadrant of the Axes (or figure):

            - loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5)
                A 2-tuple (x, y) places the corner of the legend specified by loc at x, y. For example,
                to put the legend's upper right-hand corner in the center of the Axes (or figure)
                the following keywords can be used: loc='upper right', bbox_to_anchor=(0.5, 0.5)

            - ncolsint, default: 1
                The number of columns that the legend has.
                For backward compatibility, the spelling ncol is also supported but it is discouraged.
                If both are given, ncols takes precedence.


            - fontsize : int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
                The font size of the legend. If the value is numeric the size will be the absolute font size in points.
                String values are relative to the current default font size. This argument is only used if prop
                is not specified.

            - labelcolor : str or list, default: rcParams["legend.labelcolor"] (default: 'None')
                The color of the text in the legend. Either a valid color string (for example, 'red'),
                or a list of color strings. The labelcolor can also be made to match the color of the line or marker using
                'linecolor', 'markerfacecolor' (or 'mfc'), or 'markeredgecolor' (or 'mec').
                Labelcolor can be set globally using rcParams["legend.labelcolor"] (default: 'None').
                If None, use rcParams["text.color"] (default: 'black').

            - numpointsint, default: rcParams["legend.numpoints"] (default: 1)
                The number of marker points in the legend when creating a legend entry for a Line2D (line).

            - scatterpointsint, default: rcParams["legend.scatterpoints"] (default: 1)
                The number of marker points in the legend when creating a legend entry for a PathCollection (scatter plot).

            - scatteryoffsets : iterable of floats, default: [0.375, 0.5, 0.3125]
                The vertical offset (relative to the font size) for the markers created for a scatter plot legend entry.
                0.0 is at the base the legend text, and 1.0 is at the top. To draw all markers at the same height,
                set to [0.5].

            - markerscalefloat, default: rcParams["legend.markerscale"] (default: 1.0)
                The relative size of legend markers compared to the originally drawn ones.

            - markerfirstbool, default: True If True, legend marker is placed to the left of the legend label.
                If False, legend marker is placed to the right of the legend label.

            - reversebool, default: False  If True, the legend labels are displayed in reverse order from the input.
                If False, the legend labels are displayed in the same order as the input.
                Added in version 3.7.

            - frameonbool, default: rcParams["legend.frameon"] (default: True)
                Whether the legend should be drawn on a patch (frame).

            - fancyboxbool, default: rcParams["legend.fancybox"] (default: True)
                Whether round edges should be enabled around the FancyBboxPatch which makes up the legend's background.

            - shadowNone, bool or dict, default: rcParams["legend.shadow"] (default: False)
                Whether to draw a shadow behind the legend. The shadow can be configured using Patch keywords.
                Customization via rcParams["legend.shadow"] (default: False) is currently not supported.

            - framealpha float, default: rcParams["legend.framealpha"] (default: 0.8)
                The alpha transparency of the legend's background. If shadow is activated and framealpha is None,
                the default value is ignored.

            - facecolor "inherit" or color, default: rcParams["legend.facecolor"] (default: 'inherit')
                The legend's background color. If "inherit", use rcParams["axes.facecolor"] (default: 'white').

            - edgecolor "inherit" or color, default: rcParams["legend.edgecolor"] (default: '0.8')
                The legend's background patch edge color. If "inherit", use rcParams["axes.edgecolor"] (default: 'black').

            - mode : {"expand", None}
                If mode is set to "expand" the legend will be horizontally expanded to fill the Axes area
                (or bbox_to_anchor if defines the legend's size).

            - bbox_transformNone or Transform  The transform for the bounding box (bbox_to_anchor).
                For a value of None (default) the Axes' transAxes transform will be used.

            - titlestr or None :  The legend's title. Default is no title (None).

            - title_fontproperties : None or FontProperties or dict
                The font properties of the legend's title. If None (default), the title_fontsize argument will be used
                if present; if title_fontsize is also None, the current rcParams["legend.title_fontsize"]
                (default: None) will be used.

            - title_fontsize int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'},
                default: rcParams["legend.title_fontsize"] (default: None)
                The font size of the legend's title. Note: This cannot be combined with title_fontproperties.
                If you want to set the fontsize alongside other font properties, use the size parameter
                in title_fontproperties.

            - alignment{'center', 'left', 'right'}, default: 'center'
                The alignment of the legend title and the box of entries. The entries are aligned as a single block,
                so that markers always lined up.

            - borderpad float, default: rcParams["legend.borderpad"] (default: 0.4)
                The fractional whitespace inside the legend border, in font-size units.

            - labelspacing  float, default: rcParams["legend.labelspacing"] (default: 0.5)
                The vertical space between the legend entries, in font-size units.

            - handlelength float, default: rcParams["legend.handlelength"] (default: 2.0)
                The length of the legend handles, in font-size units.

            - handleheight float, default: rcParams["legend.handleheight"] (default: 0.7)
                The height of the legend handles, in font-size units.

            - handletextpad float, default: rcParams["legend.handletextpad"] (default: 0.8)
                The pad between the legend handle and text, in font-size units.

            - borderaxespad float, default: rcParams["legend.borderaxespad"] (default: 0.5)
                The pad between the Axes and legend border, in font-size units.

            - columnspacing float, default: rcParams["legend.columnspacing"] (default: 2.0)
                The spacing between columns, in font-size units.

            - draggablebool, default: False
                Whether the legend can be dragged with the mouse.

        Returns
        -------
        None

        See Also
        --------
        matplotlib.pyplot.legend

        """
        self.param_legende.update(kwargs)

    def config_labels_contours(self, **kwargs) -> None:
        """
        
        Additionals configurations for the contours labels
        
        Parameters
        ----------
        kwargs
            
            - fontsize str or float, default: rcParams["font.size"] (default: 10.0)
                Size in points or relative size e.g., 'smaller', 'x-large'.
                See plt.Text.set_size for accepted string values.

            - colorscolor or colors or None, default: None
                The label colors:

                    If None, the color of each label matches the color of the corresponding contour.

                    If one string color, e.g., colors = 'r' or colors = 'red', all labels will be plotted in this color.

                    If a tuple of colors (string, float, RGB, etc), different labels will be plotted in different
                     colors in the order specified.

            - inlinebool, default: True
                If True the underlying contour is removed where the label is placed.

            - inline_spacingfloat, default: 5
                Space in pixels to leave on each side of label when placing inline.
                This spacing will be exact for labels at locations where the contour is straight, less so for labels on
                curved contours.

            - fmt str, optional
                How the levels are formatted:  it is interpreted as a %-style format string.
                The default is to use a standard ScalarFormatter.

            - manual bool or iterable, default: False
                If True, contour labels will be placed manually using mouse clicks. Click the first button near a
                contour to add a label, click the second button (or potentially both mouse buttons at once) to finish
                adding labels. The third button can be used to remove the last label added, but only if labels are not
                inline. Alternatively, the keyboard can be used to select label locations (enter to end label
                placement, delete or backspace act like the third mouse button, and any other key will select a label
                location).
                manual can also be an iterable object of (x, y) tuples. Contour labels will be created as if mouse is
                clicked at each (x, y) position.

            - rightside_upbool, default: True
                If True, label rotations will always be plus or minus 90 degrees from level.

            - use_clabeltext bool, default: False
                If True, use Text.set_transform_rotates_text to ensure that label rotation is updated whenever the Axes
                aspect changes.

            - zorder float or None, default: (2 + contour.get_zorder())
                zorder of the contour labels.
        
        Returns
        -------
        None
        
        See Also
        --------
        matplotlib.axes.Axes.clabel

        """
        self.param_labels_contours.update(kwargs)

    def config_fig(self, **kwargs) -> None:
        """

        Additionnals parameters to configure the Figure

        Parameters
        ----------
        kwargs
            - figsize2 : tuple of floats, default: rcParams["figure.figsize"] (default: [6.4, 4.8])
                Figure dimension (width, height) in inches.
            - dpi float, default: rcParams["figure.dpi"] (default: 100.0)
                Dots per inch.
            - facecolor default: rcParams["figure.facecolor"] (default: 'white')
                The figure patch facecolor.
            - edgecolor default: rcParams["figure.edgecolor"] (default: 'white')
                The figure patch edge color.
            - linewidthfloat
                The linewidth of the frame (i.e. the edge linewidth of the figure patch).

            - frameonbool, default: rcParams["figure.frameon"] (default: True)
                If False, suppress drawing the figure background patch.

            - layout {'constrained', 'compressed', 'tight', 'none', LayoutEngine, None}, default: None

                The layout mechanism for positioning of plot elements to avoid overlapping Axes decorations
                 (labels, ticks, etc). Note that layout managers can have significant performance penalties.

                    'constrained': The constrained layout solver adjusts Axes sizes to avoid overlapping Axes decorations.
                    Can handle complex plot layouts and colorbars, and is thus recommended.

                    See Constrained layout guide for examples.

                    'compressed': uses the same algorithm as 'constrained', but removes extra space between
                     fixed-aspect-ratio Axes. Best for simple grids of Axes.

                    'tight': Use the tight layout mechanism. This is a relatively simple algorithm that adjusts the subplot
                    parameters so that decorations do not overlap.

                    See Tight layout guide for examples.

                    'none': Do not use a layout engine.

                    A LayoutEngine instance. Builtin layout classes are ConstrainedLayoutEngine and TightLayoutEngine,
                    more easily accessible by 'constrained' and 'tight'. Passing an instance allows third parties to
                    provide their own layout engine.

                If not given, fall back to using the parameters tight_layout and constrained_layout, including their config
                defaults rcParams["figure.autolayout"] (default: False) and rcParams["figure.constrained_layout.use"]
                (default: False).

            - alpha scalar or None

            - animated bool

            - clip_on bool

            - constrained_layout unknown

            - constrained_layout_pads unknown

            - dpi float

            - edgecolor color

            - facecolor color

            - figheight float

            - figwidth float

            - frameon bool

            - gid str

            - in_layout bool

            - layout_engine  {'constrained', 'compressed', 'tight', 'none', LayoutEngine, None}

            - linewidth number

            - mouseover bool

            - picker None or bool or float or callable

            - rasterized bool

            - size_inches  (float, float) or float

            - sketch_params (scale: float, length: float, randomness: float)

            - snap bool or None

            - tight_layout

            - url str

            - visible bool

            - zorder float

        Returns
        -------
        None

        See Also
        --------

        matplotlib.pyplot.Figure

        """
        self.param_fig.update(kwargs)

    def config_enrg_fig(self, **kwargs) -> None:
        """

        Additionals parameters for the Figure saving

        Parameters
        ----------
        kwargs

            - figsize 2-tuple of floats, default: rcParams["figure.figsize"] (default: [6.4, 4.8])
                Figure dimension (width, height) in inches.

            - dpi float, default: rcParams["figure.dpi"] (default: 100.0)
                Dots per inch.

            - facecolor default: rcParams["figure.facecolor"] (default: 'white')
                The figure patch facecolor.

            - edgecolor default: rcParams["figure.edgecolor"] (default: 'white')
                The figure patch edge color.

            - linewidth float
                The linewidth of the frame (i.e. the edge linewidth of the figure patch).

            - frameon bool, default: rcParams["figure.frameon"] (default: True)
                If False, suppress drawing the figure background patch.

            - layout {'onstrained', 'compressed', 'tight', 'none', LayoutEngine, None}, default: None

                The layout mechanism for positioning of plot elements to avoid overlapping Axes decorations
                 (labels, ticks, etc). Note that layout managers can have significant performance penalties.

                    'constrained': The constrained layout solver adjusts Axes sizes to avoid overlapping Axes
                     decorations. Can handle complex plot layouts and colorbars, and is thus recommended.

                    See Constrained layout guide for examples.

                    'compressed': uses the same algorithm as 'constrained', but removes extra space between
                    fixed-aspect-ratio Axes. Best for simple grids of Axes.

                    'tight': Use the tight layout mechanism. This is a relatively simple algorithm that adjusts the
                     subplot parameters so that decorations do not overlap.

                    See Tight layout guide for examples.

                    'none': Do not use a layout engine.

                    A LayoutEngine instance. Builtin layout classes are ConstrainedLayoutEngine and TightLayoutEngine,
                     more easily accessible by 'constrained' and 'tight'. Passing an instance allows third parties to
                      provide their own layout engine.

                If not given, fall back to using the parameters tight_layout and constrained_layout, including their
                 config defaults rcParams["figure.autolayout"] (default: False) and
                 rcParams["figure.constrained_layout.use"] (default: False).

            - alpha  scalar or None

            - animated  bool

            - clip_on bool

            - constrained_layout  unknown

            - constrained_layout_pads unknown

            - dpi float

            - edgecolor color

            - facecolor  color

            - figheight float

            - figwidth float

            - frameon bool

            - gid str

            - in_layout bool

            - label object

            - layout_engine {'constrained', 'compressed', 'tight', 'none', LayoutEngine, None}

            - linewidth number

            - mouseover bool

            - picker None or bool or float

            - rasterized bool

            - size_inches (float, float) or float

            - sketch_params (scale: float, length: float, randomness: float)

            - snap bool or None

            - tight_layout unknown

            - transform  Transform

            - url str

            - visible bool

            - zorder float

        Returns
        -------
        None

        See Also
        --------

        matplotlib.figure.Figure.savefig

        """
        self.param_enrg_fig.update(kwargs)

    def config_font(self, **kwargs) -> None:
        """

        Global font parameter

        Parameters
        ----------
        kwargs
            'family' : 'fantasy','monospace','sans-serif','serif','cursive'
            'styles' : 'normal', 'italic', 'oblique'
            'size' : valeur numérique
            'variants' : 'normal', 'small-caps'
            'weight' : 'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'

        Returns
        -------

        """
        k: list[str] = kwargs.keys()
        vals: list = kwargs.values()
        kwargs: dict = {}
        for K, L in zip(k, vals):
            if "font." not in K:
                kwargs['font.' + K] = L
            else:
                kwargs[K] = L
        self.param_font.update(kwargs)

    def config_colorbar(self, index_colorbar: int = ii_max, ticks: list | np.ndarray | None = None,
                        **kwargs) -> None:
        """

        Colorbar additianal parameter

        Parameters
        ----------
        index_colorbar : int, optional
            The index of the colorbar (default the parameters are added for all colorbars)
        ticks : list[float] | array_like
            The colorbar's ticks. If None, ticks are determined automatically from the input.
        kwargs
            the parameter dictionary

        Returns
        -------
        None

        """
        if index_colorbar == ii_max:
            for d in self.param_colorbar:
                d.update(kwargs)
            if ticks is not None and len(ticks) > 0:
                self.ticks_colorbar = [ticks for t in self.ticks_colorbar]
            elif ticks is not None:
                self.ticks_colorbar = [[-np.inf] for t in self.ticks_colorbar]
        else:
            if ticks is not None and len(ticks) > 0:
                self.ticks_colorbar[index_colorbar] = ticks
            elif ticks is not None:
                self.ticks_colorbar[index_colorbar] = [-np.inf]
            self.param_colorbar[index_colorbar].update(kwargs)

    def dark_background(self) -> None:
        """

        Put a dark background on the figure

        Returns
        -------

        None

        """
        self.style = 'dark_background'
        for d in self.param_lines:
            if "color" in d.keys() and (isinstance(d["color"], str) and d["color"] == "k"):
                d["color"] = "w"
        for d in self.param_contours:
            if "color" in d.keys() and (isinstance(d["color"], str) and d["color"] == "k"):
                d["color"] = "w"
        for d in self.param_polygons:
            if "facecolor" in d.keys() and (isinstance(d["color"], str) and d["color"] == "k"):
                d["facecolor"] = "w"

    def default_style(self) -> None:
        """
        Use default style

        Returns
        -------
        None

        """
        self.style = 'default'
        for d in self.param_lines:
            if "color" in d.keys() and (isinstance(d["color"], str) and d["color"] == "w"):
                d["color"] = "k"
        if "colors" in self.param_contours.keys():
            for i in range(len(self.param_contours["colors"])):
                if (isinstance(self.param_contours["colors"][i], str)
                        and self.param_contours["colors"][i] == "w"):
                    self.param_contours["colors"] = "k"
        for d in self.param_polygons:
            if "facecolor" in d.keys() and d["facecolor"] == "w":
                d["facecolor"] = "k"

    def plot_colorbar(self) -> None:
        """
        Plot the custom colorbar(s) if it(they) exists

        Returns
        -------
        None

        """
        if self.custum_colorbar_colors is not None:
            # share_axis: np.array[int] = np.full(len(self.custum_colorbar_colors), -1)
            right: list[int] = []
            sr: np.float64 = 0.
            top: list[int] = []
            st: np.float64 = 0.
            left: list[int] = []
            sl: np.float64 = 0.
            bottom: list[int] = []
            sb: np.float64 = 0.

            size_colorbar: np.float64 = 0.02
            size_legend: np.float64 = 10 * size_colorbar
            size_space: np.float64 = 0.
            for i in range(len(self.custum_colorbar_colors)):
                if "location" not in self.param_colorbar[i] or self.param_colorbar[i]["location"] == "right":
                    right.append(i)
                    if len(right) == 1:
                        self.param_colorbar[i]["share_axis"] = True
                    if "size" in self.param_colorbar[i].keys():
                        sr += self.param_colorbar[i]["size"]
                    else:
                        sr += size_colorbar
                    if ("share_axis" not in self.param_colorbar[i].keys()
                            or not self.param_colorbar[i]["share_axis"]):
                        if "size_legend" in self.param_colorbar[i].keys():
                            sr += self.param_colorbar[i]["size_legend"]
                        elif "size" in self.param_colorbar[i].keys():
                            sr += self.param_colorbar[i]["size"] * size_legend / size_colorbar
                        else:
                            sr += size_legend
                    if "space_between" in self.param_colorbar[i].keys():
                        sr += self.param_colorbar[i]["space_between"]
                    else:
                        sr += size_space
                elif self.param_colorbar[i]["location"] == "top":
                    top.append(i)
                    if len(top) == 1:
                        self.param_colorbar[i]["share_axis"] = True
                    if "size" in self.param_colorbar[i].keys():
                        st += self.param_colorbar[i]["size"]
                    else:
                        st += size_colorbar
                    if ("share_axis" not in self.param_colorbar[i].keys()
                            or not self.param_colorbar[i]["share_axis"]):
                        if "size_legend" in self.param_colorbar[i].keys():
                            st += self.param_colorbar[i]["size_legend"]
                        elif "size" in self.param_colorbar[i].keys():
                            st += self.param_colorbar[i]["size"] * size_legend / size_colorbar
                        else:
                            st += size_legend
                    if "space_between" in self.param_colorbar[i].keys():
                        st += self.param_colorbar[i]["space_between"]
                    else:
                        st += size_space
                elif self.param_colorbar[i]["location"] == "left":
                    left.append(i)
                    if len(left) == 1:
                        self.param_colorbar[i]["share_axis"] = True
                    if "size" in self.param_colorbar[i].keys():
                        sl += self.param_colorbar[i]["size"]
                    else:
                        sl += size_colorbar
                    if ("share_axis" not in self.param_colorbar[i].keys()
                            or not self.param_colorbar[i]["share_axis"]):
                        if "size_legend" in self.param_colorbar[i].keys():
                            sl += self.param_colorbar[i]["size_legend"]
                        elif "size" in self.param_colorbar[i].keys():
                            sl += self.param_colorbar[i]["size"] * size_legend / size_colorbar
                        else:
                            sl += size_legend
                    if "space_between" in self.param_colorbar[i].keys():
                        sl += self.param_colorbar[i]["space_between"]
                    else:
                        sl += size_space
                elif self.param_colorbar[i]["location"] == "bottom":
                    bottom.append(i)
                    if len(bottom) == 1:
                        self.param_colorbar[i]["share_axis"] = True
                    if "size" in self.param_colorbar[i].keys():
                        sb += self.param_colorbar[i]["size"]
                    else:
                        sb += size_colorbar
                    if ("share_axis" not in self.param_colorbar[i].keys()
                            or not self.param_colorbar[i]["share_axis"]):
                        if "size_legend" in self.param_colorbar[i].keys():
                            sb += self.param_colorbar[i]["size_legend"]
                        elif "size" in self.param_colorbar[i].keys():
                            sb += self.param_colorbar[i]["size"] * size_legend / size_colorbar
                        else:
                            sb += size_legend
                    if "space_between" in self.param_colorbar[i].keys():
                        sb += self.param_colorbar[i]["space_between"]
                    else:
                        sb += size_space
                else:
                    raise UserWarning("Graphique.plot_colorbar : the location ", self.param_colorbar[i]["location"],
                                      "isn't awalible. Please use right, top, left, bottom")

            pos_r: np.float64 = 1.
            pos_t: np.float64 = 1.
            pos_l: np.float64 = 0.
            pos_b: np.float64 = 0.
            for (i, ii) in zip(right, np.arange(len(right))):
                cmap = mpl.colors.ListedColormap(self.custum_colorbar_colors[i])
                norm = mpl.colors.BoundaryNorm(self.custum_colorbar_values[i], cmap.N)
                params: dict = self.param_colorbar[i].copy()
                fmt: str = ""
                if "format" in params.keys():
                    fmt = params["format"]
                    del params["format"]
                ticks_labels = None
                if "ticks_labels" in params.keys():
                    ticks_labels = params["ticks_label"]
                    del params["ticks_label"]
                elif fmt != "" and len(self.ticks_colorbar[i]) != 0 and self.ticks_colorbar[i][0] > - np.inf:
                    ticks_labels = [fmt.format(x) for x in self.ticks_colorbar[i]]
                scale: str = "linear"
                if "scale" in params.keys():
                    scale = params["scale"]
                    del params["scale"]
                share_axis: bool = False
                if "share_axis" in params.keys():
                    share_axis = params["share_axis"]
                    del params["share_axis"]
                if "location" in params.keys():
                    del params["location"]
                fraction: np.float64 = 1
                if "fraction" in params.keys():
                    fraction = params["fraction"]
                    del params["fraction"]

                size_cb: np.float64 = size_colorbar
                if "size" in params.keys():
                    size_cb = params["size"]
                    del params["size"]
                size: np.float64 = size_cb
                if not share_axis and "size_legend" in params.keys():
                    size += params["size_legend"]
                    del params["size_legend"]
                elif not share_axis:
                    size += size_legend
                if "space_between" in params.keys():
                    size += params["space_between"]
                    del params["space_between"]
                elif not share_axis:
                    size += size_space
                cax = self.ax.inset_axes([pos_r - size, 0.5 - fraction / 2.,
                                          size_cb, fraction])
                # cax = self.fig.add_axes([pos_r - size, 0.5 - fraction / 2.,
                #                           size_cb, fraction])
                self.axes.append(cax)
                pos_r -= size
                # cax.set_xticks([])
                # cax.set_yticks([])
                # cax.set_axis_off()

                # if share_axis:
                #     cax.sharey = self.colorbar[-1].ax
                self.colorbar.append(self.fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                                                       cax=cax, **params))
                self.cmap.append(cmap)
                self.norms.append(norm)
                if len(self.ticks_colorbar[i]) == 0 and scale != "linear":
                    self.colorbar[-1].ax.set_yscale(scale)
                elif (ticks_labels is not None and len(self.ticks_colorbar[i]) > 0
                      and self.ticks_colorbar[i][0] > - np.inf):
                    self.colorbar[-1].set_ticks(ticks=self.ticks_colorbar[i], labels=ticks_labels)
                elif len(self.ticks_colorbar[i]) > 0 and self.ticks_colorbar[i][0] > - np.inf:
                    self.colorbar[-1].set_ticks(ticks=self.ticks_colorbar[i])
                elif len(self.ticks_colorbar[i]) > 0 and self.ticks_colorbar[i][0] == - np.inf:
                    self.colorbar[-1].set_ticks(ticks=[])

            for (i, ii) in zip(top, np.arange(len(right))):
                cmap = mpl.colors.ListedColormap(self.custum_colorbar_colors[i])
                norm = mpl.colors.BoundaryNorm(self.custum_colorbar_values[i], cmap.N)
                params: dict = self.param_colorbar[i].copy()
                fmt: str = ""
                if "format" in params.keys():
                    fmt = params["format"]
                    del params["format"]
                ticks_labels = None
                if "ticks_labels" in params.keys():
                    ticks_labels = params["ticks_label"]
                    del params["ticks_label"]
                elif fmt != "" and len(self.ticks_colorbar[i]) != 0 and self.ticks_colorbar[i][0] > - np.inf:
                    ticks_labels = [fmt.format(x) for x in self.ticks_colorbar[i]]
                scale: str = "linear"
                if "scale" in params.keys():
                    scale = params["scale"]
                    del params["scale"]
                share_axis: bool = False
                if "share_axis" in params.keys():
                    share_axis = params["share_axis"]
                    del params["share_axis"]
                if "location" in params.keys():
                    del params["location"]

                fraction: np.float64 = 1.
                if "fraction" in params.keys():
                    fraction = params["fraction"]
                    del params["fraction"]

                size_cb: np.float64 = size_colorbar
                if "size" in params.keys():
                    size_cb = params["size"]
                    del params["size"]
                size: np.float64 = size_cb
                if not share_axis and "size_legend" in params.keys():
                    size += params["size_legend"]
                    del params["size_legend"]
                elif not share_axis:
                    size += size_legend
                if "space_between" in params.keys():
                    size += params["space_between"]
                    del params["space_between"]
                elif not share_axis:
                    size += size_space
                cax = self.ax.inset_axes([0.5 - fraction / 2., pos_t - size,
                                          fraction, size_cb])
                # cax = self.fig.add_axes([0.5 - fraction / 2., pos_t - size,
                #                           fraction, size_cb])
                self.axes.append(cax)
                pos_t -= size
                # cax.set_xticks([])
                # cax.set_yticks([])
                # cax.set_axis_off()

                # if share_axis:
                #     cax.sharey = self.colorbar[-1].ax
                self.colorbar.append(self.fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                                                       cax=cax, **params))
                self.cmap.append(cmap)
                self.norms.append(norm)
                if len(self.ticks_colorbar[i]) == 0 and scale != "linear":
                    self.colorbar[-1].ax.set_yscale(scale)
                elif (ticks_labels is not None and len(self.ticks_colorbar[i]) > 0
                      and self.ticks_colorbar[i][0] > - np.inf):
                    self.colorbar[-1].set_ticks(ticks=self.ticks_colorbar[i], labels=ticks_labels)
                elif len(self.ticks_colorbar[i]) > 0 and self.ticks_colorbar[i][0] > - np.inf:
                    self.colorbar[-1].set_ticks(ticks=self.ticks_colorbar[i])
                elif len(self.ticks_colorbar[i]) > 0 and self.ticks_colorbar[i][0] == - np.inf:
                    self.colorbar[-1].set_ticks(ticks=[])

            for (i, ii) in zip(left, np.arange(len(right))):
                cmap = mpl.colors.ListedColormap(self.custum_colorbar_colors[i])
                norm = mpl.colors.BoundaryNorm(self.custum_colorbar_values[i], cmap.N)
                params: dict = self.param_colorbar[i].copy()
                fmt: str = ""
                if "format" in params.keys():
                    fmt = params["format"]
                    del params["format"]
                ticks_labels = None
                if "ticks_labels" in params.keys():
                    ticks_labels = params["ticks_label"]
                    del params["ticks_label"]
                elif fmt != "" and len(self.ticks_colorbar[i]) != 0 and self.ticks_colorbar[i][
                    0] > - np.inf:
                    ticks_labels = [fmt.format(x) for x in self.ticks_colorbar[i]]
                scale: str = "linear"
                if "scale" in params.keys():
                    scale = params["scale"]
                    del params["scale"]
                share_axis: bool = False
                if "share_axis" in params.keys():
                    share_axis = params["share_axis"]
                    del params["share_axis"]
                if "location" in params.keys():
                    del params["location"]

                fraction: np.float64 = 1
                if "fraction" in params.keys():
                    fraction = params["fraction"]
                    del params["fraction"]

                size_cb: np.float64 = size_colorbar
                if "size" in params.keys():
                    size_cb = params["size"]
                    del params["size"]
                size: np.float64 = size_cb
                if not share_axis and "size_legend" in params.keys():
                    size += params["size_legend"]
                    del params["size_legend"]
                elif not share_axis:
                    size += size_legend

                if "space_between" in params.keys():
                    size += params["space_between"]
                    del params["space_between"]
                elif not share_axis:
                    size += size_space

                cax = self.ax.inset_axes([pos_l, 0.5 - fraction / 2., size_cb, fraction])
                # cax = self.fig.add_axes([pos_l, 0.5 - fraction / 2., size_cb, fraction])
                self.axes.append(cax)
                pos_l += size
                # cax.set_xticks([])
                # cax.set_yticks([])
                # cax.set_axis_off()

                # if share_axis:
                #     cax.sharey = self.colorbar[-1].ax
                self.colorbar.append(self.fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                                                       cax=cax, **params))
                self.cmap.append(cmap)
                self.norms.append(norm)
                if len(self.ticks_colorbar[i]) == 0 and scale != "linear":
                    self.colorbar[-1].ax.set_yscale(scale)
                elif (ticks_labels is not None and len(self.ticks_colorbar[i]) > 0
                      and self.ticks_colorbar[i][0] > - np.inf):
                    self.colorbar[-1].set_ticks(ticks=self.ticks_colorbar[i], labels=ticks_labels)
                elif len(self.ticks_colorbar[i]) > 0 and self.ticks_colorbar[i][0] > - np.inf:
                    self.colorbar[-1].set_ticks(ticks=self.ticks_colorbar[i])
                elif len(self.ticks_colorbar[i]) > 0 and self.ticks_colorbar[i][0] == - np.inf:
                    self.colorbar[-1].set_ticks(ticks=[])

            for (i, ii) in zip(bottom, np.arange(len(right))):
                cmap = mpl.colors.ListedColormap(self.custum_colorbar_colors[i])
                norm = mpl.colors.BoundaryNorm(self.custum_colorbar_values[i], cmap.N)
                params: dict = self.param_colorbar[i].copy()
                fmt: str = ""
                if "format" in params.keys():
                    fmt = params["format"]
                    del params["format"]
                ticks_labels = None
                if "ticks_labels" in params.keys():
                    ticks_labels = params["ticks_label"]
                    del params["ticks_label"]
                elif fmt != "" and len(self.ticks_colorbar[i]) != 0 and self.ticks_colorbar[i][0] > - np.inf:
                    ticks_labels = [fmt.format(x) for x in self.ticks_colorbar[i]]
                scale: str = "linear"
                if "scale" in params.keys():
                    scale = params["scale"]
                    del params["scale"]
                share_axis: bool = False
                if "share_axis" in params.keys():
                    share_axis = params["share_axis"]
                    del params["share_axis"]
                if "location" in params.keys():
                    del params["location"]

                fraction: np.float64 = np.double(1.)
                if "fraction" in params.keys():
                    fraction = params["fraction"]
                    del params["fraction"]

                size_cb: np.float64 = size_colorbar
                if "size" in params.keys():
                    size_cb = params["size"]
                    del params["size"]
                size: np.float64 = size_cb
                if not share_axis and "size_legend" in params.keys():
                    size += params["size_legend"]
                    del params["size_legend"]
                elif not share_axis:
                    size += size_legend
                if "space_between" in params.keys():
                    size += params["space_between"]
                    del params["space_between"]
                elif not share_axis:
                    size += size_space
                cax = self.ax.inset_axes([0.5 - fraction / 2., pos_b, size_cb, fraction])
                # cax = self.fig.add_axes([0.5 - fraction / 2., pos_b, size_cb, fraction])
                self.axes.append(cax)
                pos_b += size
                # cax.set_xticks([])
                # cax.set_yticks([])
                # cax.set_axis_off()

                # if share_axis:
                #     cax.sharey = self.colorbar[-1].ax
                self.colorbar.append(self.fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                                                       cax=cax, **params))
                self.cmap.append(cmap)
                self.norms.append(norm)
                if len(self.ticks_colorbar[i]) == 0 and scale != "linear":
                    self.colorbar[-1].ax.set_yscale(scale)
                elif (ticks_labels is not None and len(self.ticks_colorbar[i]) > 0
                      and self.ticks_colorbar[i][0] > - np.inf):
                    self.colorbar[-1].set_ticks(ticks=self.ticks_colorbar[i], labels=ticks_labels)
                elif len(self.ticks_colorbar[i]) > 0 and self.ticks_colorbar[i][0] > - np.inf:
                    self.colorbar[-1].set_ticks(ticks=self.ticks_colorbar[i])
                elif len(self.ticks_colorbar[i]) > 0 and self.ticks_colorbar[i][0] == - np.inf:
                    self.colorbar[-1].set_ticks(ticks=[])

            if len(right) > 0:
                if "space_between" in self.param_colorbar[right[-1]].keys():
                    sr += self.param_colorbar[right[-1]]["space_between"]
                elif "size" in self.param_colorbar[right[-1]].keys():
                    sr += self.param_colorbar[right[-1]]["size"] / 2.
                else:
                    sr += size_colorbar / 2
            if len(top) > 0:
                if "space_between" in self.param_colorbar[top[-1]].keys():
                    st += self.param_colorbar[top[-1]]["space_between"]
                elif "size" in self.param_colorbar[top[-1]].keys():
                    st += self.param_colorbar[top[-1]]["size"] / 2.
                else:
                    st += size_colorbar / 2.
            if len(left) > 0:
                if "space_between" in self.param_colorbar[left[-1]].keys():
                    sl += self.param_colorbar[left[-1]]["space_between"]
                    pos_l += self.param_colorbar[left[-1]]["space_between"]
                elif "size" in self.param_colorbar[left[-1]].keys():
                    sl += self.param_colorbar[left[-1]]["size"] / 2.
                    pos_l += self.param_colorbar[left[-1]]["size"] / 2.
                else:
                    sl += size_colorbar / 2
                    pos_l += size_colorbar / 2
            if len(bottom) > 0:
                if "space_between" in self.param_colorbar[bottom[-1]].keys():
                    sb += self.param_colorbar[bottom[-1]]["space_between"]
                    pos_b += self.param_colorbar[bottom[-1]]["space_between"]
                elif "size" in self.param_colorbar[bottom[-1]].keys():
                    sb += self.param_colorbar[bottom[-1]]["size"] / 2.
                    pos_b += self.param_colorbar[bottom[-1]]["size"] / 2.
                else:
                    sb += size_colorbar / 2
                    pos_b += size_colorbar / 2
            self.ax.set_axis_off()
            self.ax.set_navigate(False)
            new_ax_ = self.ax.inset_axes([pos_l, pos_b, 1. - (sl + sr), 1. - (st + sb)])
            new_ax_.set_axis_off()
            new_ax = self.fig.add_axes(new_ax_.get_position())
            # new_ax = self.fig.add_axes([pos_l, pos_b, 1. - (sl + sr), 1. - (st + sb)])
            new_ax.set_navigate(True)
            self.axes.append(new_ax)
            new_ax.sharex = self.ax
            new_ax.sharey = self.ax
            self.ax = new_ax
            self.ax.set_navigate(True)

    def plot_lines(self) -> None:
        """
        Plot all the line of the Graphique

        Returns
        -------
        None

        """
        with mpl.rc_context(self.param_font):
            if self.ax is None:
                self.ax = self.fig.add_axes([0, 0, 1, 1], **self.param_ax)
                if self.axes is None:
                    self.axes = [self.ax]
                else:
                    self.axes.append(self.ax)
                if self.ax_tl is None:
                    self.ax_tl = self.ax.twiny()
                    self.ax_tl.set(**self.param_ax_tl)
                if self.ax_br is None:
                    self.ax_br = self.ax.twinx()
                    self.ax_br.set(**self.param_ax_br)
                if self.ax_tr is None:
                    self.ax_tr = self.ax_tl.twinx()
                    self.ax_tr.set(**self.param_ax_tr)

            index: np.ndarray[int] = np.arange(0, len(self.lines_x))
            if len(self.indexs_plot_lines) > 0:
                index = np.array(self.indexs_plot_lines)
            for i in index:
                if len(self.param_lines[i]) > 0:
                    params_i = self.param_lines[i].copy()
                    if "axis_config" not in params_i.keys():
                        axis_config: str = "bl"
                    else:
                        axis_config: str = params_i["axis_config"]
                        del params_i["axis_config"]
                    if len(self.err_y[i]) > 0:
                        if isinstance(self.err_x[i], list | np.ndarray) and len(self.err_x[i]) == 0:
                            err_x = None
                        else:
                            err_x = self.err_x[i]
                        if axis_config == "bl":
                            self.ax.errorbar(
                                x=self.lines_x[i], y=self.lines_y[i], xerr=err_x,
                                yerr=self.err_y[i], **params_i)
                        elif axis_config == "tl":
                            self.ax_tl.errorbar(
                                x=self.lines_x[i], y=self.lines_y[i], xerr=err_x,
                                yerr=self.err_y[i], **params_i)
                        elif axis_config == "tr":
                            self.ax_tr.errorbar(
                                x=self.lines_x[i], y=self.lines_y[i], xerr=err_x,
                                yerr=self.err_y[i], **params_i)
                        else:  # axis_config == "br"
                            self.ax_br.errorbar(
                                x=self.lines_x[i], y=self.lines_y[i], xerr=err_x,
                                yerr=self.err_y[i], **params_i)

                    else:
                        if "color" in params_i and isinstance(params_i["color"],
                                                              list | np.ndarray):
                            marker: str = ""
                            if "marker" in params_i.keys():
                                marker = params_i["marker"]
                                del params_i["marker"]

                            if axis_config == "bl":
                                self.ax.scatter(
                                    self.lines_x[i], self.lines_y[i], marker=marker, **params_i)
                            elif axis_config == "tl":
                                self.ax_tl.scatter(
                                    self.lines_x[i], self.lines_y[i], marker=marker, **params_i)
                            elif axis_config == "tr":
                                self.ax_tr.scatter(
                                    self.lines_x[i], self.lines_y[i], marker=marker, **params_i)
                            else:  # axis_config == "br":
                                self.ax_br.scatter(
                                    self.lines_x[i], self.lines_y[i], marker=marker, **params_i)

                            if not ("linestyle" in params_i.keys()) or params_i["linestyle"] != "":
                                colors = [to_rgba(c) for c in params_i["color"]]

                                del params_i["color"]
                                if axis_config == "bl":
                                    colored_line(self.lines_x[i], self.lines_y[i], colors, self.ax, **params_i)
                                elif axis_config == "tl":
                                    colored_line(self.lines_x[i], self.lines_y[i], colors, self.ax_tl, **params_i)
                                elif axis_config == "tr":
                                    colored_line(self.lines_x[i], self.lines_y[i], colors, self.ax_tr, **params_i)
                                else:  # axis_config == "tr"
                                    colored_line(self.lines_x[i], self.lines_y[i], colors, self.ax_br, **params_i)

                        else:
                            if axis_config == "bl":
                                self.ax.plot(self.lines_x[i], self.lines_y[i], **params_i)
                            elif axis_config == "tl":
                                self.ax_tl.plot(self.lines_x[i], self.lines_y[i], **params_i)
                            elif axis_config == "tr":
                                self.ax_tr.plot(self.lines_x[i], self.lines_y[i], **params_i)
                            else:  # axis_config == "br"
                                self.ax_br.plot(self.lines_x[i], self.lines_y[i], **params_i)
                else:
                    self.ax.plot(self.lines_x[i], self.lines_y[i])

            # bl, default axis
            if len(self.x_axe[0]) == 0 or self.x_axe[0][0] > -np.inf:
                if len(self.x_axe[0]) > 0:
                    self.ax.set_xlim([self.x_axe[0].min(), self.x_axe[0].max()])
                self.ax.set_xticks(self.x_axe[0])
            if len(self.labels_x_ticks[0]) == 0 or self.labels_x_ticks[0][0] != "empty":
                self.ax.set_xticklabels(self.labels_x_ticks[0])
            if len(self.y_axe[0]) == 0 or self.y_axe[0][0] > -np.inf:
                if len(self.y_axe[0]) > 0:
                    self.ax.set_ylim([self.y_axe[0].min(), self.y_axe[0].max()])
                self.ax.set_yticks(self.y_axe[0])
            if len(self.labels_y_ticks[0]) == 0 or self.labels_y_ticks[0][0] != "empty":
                self.ax.set_yticklabels(self.labels_y_ticks[0])

                # tl
            if len(self.x_axe[1]) == 0 or self.x_axe[1][0] > -np.inf:
                if len(self.x_axe[1]) > 0:
                    self.ax_tl.set_xlim([self.x_axe[1].min(), self.x_axe[1].max()])
                self.ax_tl.set_xticks(self.x_axe[1])
            if len(self.labels_x_ticks[1]) == 0 or self.labels_x_ticks[1][0] != "empty":
                self.ax_tl.set_xticklabels(self.labels_x_ticks[1])
            if len(self.y_axe[1]) == 0 or self.y_axe[1][0] > -np.inf:
                if len(self.y_axe[1]) > 0:
                    self.ax_tl.set_ylim([self.y_axe[1].min(), self.y_axe[1].max()])
                self.ax_tl.set_yticks(self.y_axe[1])
            if len(self.labels_y_ticks[1]) == 0 or self.labels_y_ticks[1][0] != "empty":
                self.ax_tl.set_yticklabels(self.labels_y_ticks[1])

                # tr
            if len(self.x_axe[2]) == 0 or self.x_axe[2][0] > -np.inf:
                if len(self.x_axe[2]) > 0:
                    self.ax_tr.set_xlim([self.x_axe[2].min(), self.x_axe[2].max()])
                self.ax_tr.set_xticks(self.x_axe[2])
            if len(self.labels_x_ticks[2]) == 0 or self.labels_x_ticks[2][0] != "empty":
                self.ax_tr.set_xticklabels(self.labels_x_ticks[2])
            if len(self.y_axe[2]) == 0 or self.y_axe[2][0] > -np.inf:
                if len(self.y_axe[2]) > 0:
                    self.ax_tr.set_ylim([self.y_axe[2].min(), self.y_axe[2].max()])
                self.ax_tr.set_yticks(self.y_axe[2])
            if len(self.labels_y_ticks[2]) == 0 or self.labels_y_ticks[2][0] != "empty":
                self.ax_tr.set_yticklabels(self.labels_y_ticks[2])

                # br
            if len(self.x_axe[3]) == 0 or self.x_axe[3][0] > -np.inf:
                if len(self.x_axe[3]) > 0:
                    self.ax_br.set_xlim([self.x_axe[3].min(), self.x_axe[3].max()])
                self.ax_br.set_xticks(self.x_axe[3])
            if len(self.labels_x_ticks[3]) == 0 or self.labels_x_ticks[3][0] != "empty":
                self.ax_br.set_xticklabels(self.labels_x_ticks[3])
            if len(self.y_axe[3]) == 0 or self.y_axe[3][0] > -np.inf:
                if len(self.y_axe[3]) > 0:
                    self.ax_br.set_ylim([self.y_axe[3].min(), self.y_axe[3].max()])
                self.ax_br.set_yticks(self.y_axe[3])
            if len(self.labels_y_ticks[3]) == 0 or self.labels_y_ticks[3][0] != "empty":
                self.ax_br.set_yticklabels(self.labels_y_ticks[3])

            if self.title != "":
                self.ax.set_title(self.title)

    def plot_texts(self) -> None:
        """
        Plot all the texts of the Graphique

        Returns
        -------
        None

        """
        with mpl.rc_context(self.param_font):
            if self.ax is None:
                self.ax = self.fig.add_axes([0, 0, 1, 1], **self.param_ax)
                if self.axes is None:
                    self.axes = [self.ax]
                else:
                    self.axes.append(self.ax)
                if self.ax_tl is None:
                    self.ax_tl = self.ax.twiny()
                    self.ax_tl.set(**self.param_ax_tl)
                if self.ax_br is None:
                    self.ax_br = self.ax.twinx()
                    self.ax_br.set(**self.param_ax_br)
                if self.ax_tr is None:
                    self.ax_tr = self.ax_tl.twinx()
                    self.ax_tr.set(**self.param_ax_tr)

            for i in range(len(self.lines_t_x)):
                # for (X, Y, S) in zip(self.lines_t_x[i], self.lines_t_y[i], self.lines_t_s[i]):
                params_i = self.param_texts[i].copy()
                if "axis_config" in params_i.keys():
                    axis_config: str = params_i["axis_config"]
                    del params_i["axis_config"]
                else:  # axis_config="bl"
                    axis_config: str = "bl"
                if axis_config == "bl":
                    self.ax.text(self.lines_t_x[i], self.lines_t_y[i], self.lines_t_s[i], **params_i)
                elif axis_config == "tl":
                    self.ax_tl.text(self.lines_t_x[i], self.lines_t_y[i], self.lines_t_s[i], **params_i)
                elif axis_config == "tr":
                    self.ax_tr.text(self.lines_t_x[i], self.lines_t_y[i], self.lines_t_s[i], **params_i)
                else:  # axis_config == "br"
                    self.ax_br.text(self.lines_t_x[i], self.lines_t_y[i], self.lines_t_s[i], **params_i)

    def plot_histogrammes(self) -> None:
        """
        Plot the Graphique's histogramme (if there is one)

        Returns
        -------
        None

        """
        with mpl.rc_context(self.param_font):
            # Axe creation
            if self.ax is None:
                self.ax = self.fig.add_axes([0, 0, 1, 1], **self.param_ax)
                if self.axes is None:
                    self.axes = [self.ax]
                else:
                    self.axes.append(self.ax)
                if self.ax_tl is None:
                    self.ax_tl = self.ax.twiny()
                    self.ax_tl.set(**self.param_ax_tl)
                if self.ax_br is None:
                    self.ax_br = self.ax.twinx()
                    self.ax_br.set(**self.param_ax_br)
                if self.ax_tr is None:
                    self.ax_tr = self.ax_tl.twinx()
                    self.ax_tr.set(**self.param_ax_tr)

            # plotting histogrammes
            for i in range(len(self.vals_histogramme)):
                pos = self.bords_histogramme[i][:-1]
                largeur = np.array(
                    self.bords_histogramme[i][1:]) - np.array(self.bords_histogramme[i][:-1])
                if len(self.param_histogrammes[i]) > 0:
                    self.ax.bar(x=pos, height=self.vals_histogramme[i], width=largeur, align='edge',
                                **self.param_histogrammes[i])
                else:
                    self.ax.bar(
                        x=pos, height=self.vals_histogramme[i], width=largeur, align='edge')
            # axes coordinate configuration
            if np.any([len(self.param_lines[i]) > 0 for i in range(len(self.param_lines))]):
                self.ax.legend(**self.param_legende)
            if len(self.x_axe) == 0 or self.x_axe[0] > -np.inf:
                if len(self.x_axe) > 0:
                    self.ax.set_xlim([self.x_axe.min(), self.x_axe.max()])
                self.ax.set_xticks(self.x_axe)
            if len(self.labels_x_ticks) == 0 or self.labels_x_ticks[0] != "empty":
                self.ax.set_xticklabels(self.labels_x_ticks)
            if len(self.y_axe) == 0 or self.y_axe[0] > -np.inf:
                if len(self.y_axe) > 0:
                    self.ax.set_ylim([self.y_axe.min(), self.y_axe.max()])
                self.ax.set_yticks(self.y_axe)
            if len(self.labels_y_ticks) == 0 or self.labels_y_ticks[0] != "empty":
                self.ax.set_yticklabels(self.labels_y_ticks)
            if self.title != "":
                self.ax.set_title(self.title)

    def plot_image(self) -> None:
        """
        Plot the Graphique's image (if there is one)

        Returns
        -------
        None

        """
        with mpl.rc_context(self.param_font):
            if self.ax is None:
                self.ax = self.fig.add_axes([0, 0, 1, 1], **self.param_ax)
                if self.axes is None:
                    self.axes = [self.ax]
                else:
                    self.axes.append(self.ax)
                if self.ax_tl is None:
                    self.ax_tl = self.ax.twiny()
                    self.ax_tl.set(**self.param_ax_tl)
                if self.ax_br is None:
                    self.ax_br = self.ax.twinx()
                    self.ax_br.set(**self.param_ax_br)
                if self.ax_tr is None:
                    self.ax_tr = self.ax_tl.twinx()
                    self.ax_tr.set(**self.param_ax_tr)

            params_tableau: dict = self.param_image.copy()
            if "axis_config" in params_tableau.keys():
                axis_config: str = params_tableau["axis_config"]
                del params_tableau["axis_config"]
            else:  # bl
                axis_config: str = "bl"

            if len(self.array_image.shape) == 2:
                if axis_config == "bl":
                    self.ax.pcolor(self.x_axe_image,
                                   self.y_axe_image, self.array_image,
                                   cmap=self.cmap[self.index_colorbar_image[0]],
                                   norm=self.norms[self.index_colorbar_image[0]],
                                   **params_tableau)
                elif axis_config == "tl":
                    self.ax_tl.pcolor(self.x_axe_image,
                                      self.y_axe_image, self.array_image,
                                      cmap=self.cmap[self.index_colorbar_image[0]],
                                      norm=self.norms[self.index_colorbar_image[0]],
                                      **params_tableau)
                elif axis_config == "tr":
                    self.ax_tr.pcolor(self.x_axe_image,
                                      self.y_axe_image, self.array_image,
                                      cmap=self.cmap[self.index_colorbar_image[0]],
                                      norm=self.norms[self.index_colorbar_image[0]],
                                      **params_tableau)
                else:  # axis_config == "br"
                    self.ax_br.pcolor(self.x_axe_image,
                                      self.y_axe_image, self.array_image,
                                      cmap=self.cmap[self.index_colorbar_image[0]],
                                      norm=self.norms[self.index_colorbar_image[0]],
                                      **params_tableau)
            else:
                if axis_config == "bl":
                    self.ax.imshow(self.array_image)
                elif axis_config == "tl":
                    self.ax_tl.imshow(self.array_image)
                elif axis_config == "tr":
                    self.ax_tr.imshow(self.array_image)
                else:  # axis_config == "br"
                    self.ax_br.imshow(self.array_image)

    def plot_contours(self) -> None:
        """
        Plot the Graphique's contours (if there are)

        Returns
        -------
        None

        """
        params: dict = self.param_contours.copy()
        if len(self.color_label_contours) > 0:
            params["colors"] = self.color_label_contours
        if "axis_config" in params.keys():
            axis_config: str = params["axis_config"]
            del params["axis_config"]
        else:  # bl
            axis_config: str = "bl"

        with mpl.rc_context(self.param_font):
            if self.ax is None:
                self.ax = self.fig.add_axes([0, 0, 1, 1], **self.param_ax)
                if self.axes is None:
                    self.axes = [self.ax]
                else:
                    self.axes.append(self.ax)
                if self.ax_tl is None:
                    self.ax_tl = self.ax.twiny()
                    self.ax_tl.set(**self.param_ax_tl)
                if self.ax_br is None:
                    self.ax_br = self.ax.twinx()
                    self.ax_br.set(**self.param_ax_br)
                if self.ax_tr is None:
                    self.ax_tr = self.ax_tl.twinx()
                    self.ax_tr.set(**self.param_ax_tr)

            if len(self.levels) > 0:
                levels = self.levels  # print("levels",levels)
            else:
                levels = self.nb_contours  # print("levels",levels)
            if self.tab_contours_is_image:
                if len(self.x_axe_image) != self.array_image.shape[1]:
                    x_axe = self.x_axe_image[:-1] + abs(
                        abs(self.x_axe_image[1:])
                        - abs(self.x_axe_image[:-1]))
                else:
                    x_axe = self.x_axe_image
                if len(self.y_axe_image) != self.array_image.shape[0]:
                    y_axe = self.y_axe_image[:-1] + abs(
                        abs(self.y_axe_image[1:])
                        - abs(self.y_axe_image[:-1]))
                else:
                    y_axe = self.y_axe_image
                if axis_config == "bl":
                    cs = self.ax.contour(x_axe, y_axe,
                                         self.array_image, levels, **params)
                elif axis_config == "tl":
                    cs = self.ax_tl.contour(x_axe, y_axe,
                                            self.array_image, levels, **params)
                elif axis_config == "tr":
                    cs = self.ax_tr.contour(x_axe, y_axe,
                                            self.array_image, levels, **params)
                else:  # axis_config == "br"
                    cs = self.ax_br.contour(x_axe, y_axe,
                                            self.array_image, levels, **params)
            else:
                if axis_config == "bl":
                    cs = self.ax.contour(self.x_axe_contours, self.y_axe_contours,
                                         self.array_contours, levels, **params)
                    if len(self.clabels) > 0:
                        dic_labels: dict = {}
                        for (n, l) in zip(self.levels, self.clabels):
                            dic_labels[n] = l
                        if len(self.clabels_mask) > 0:
                            self.ax.clabel(cs, self.levels[self.clabels_mask], fmt=dic_labels,
                                           **self.param_labels_contours)
                        else:
                            self.ax.clabel(cs, self.levels, fmt=dic_labels, **self.param_labels_contours)
                    else:
                        self.ax.clabel(cs, **self.param_labels_contours)

                elif axis_config == "tl":
                    cs = self.ax_tl.contour(self.x_axe_contours, self.y_axe_contours,
                                            self.array_contours, levels, **params)
                    if len(self.clabels) > 0:
                        dic_labels: dict = {}
                        for (n, l) in zip(self.levels, self.clabels):
                            dic_labels[n] = l
                        if len(self.clabels_mask) > 0:
                            self.ax_tl.clabel(cs, self.levels[self.clabels_mask], fmt=dic_labels,
                                              **self.param_labels_contours)
                        else:
                            self.ax_tl.clabel(cs, self.levels, fmt=dic_labels, **self.param_labels_contours)
                    else:
                        self.ax_tl.clabel(cs, **self.param_labels_contours)

                elif axis_config == "tr":
                    cs = self.ax_tr.contour(self.x_axe_contours, self.y_axe_contours,
                                            self.array_contours, levels, **params)
                    if len(self.clabels) > 0:
                        dic_labels: dict = {}
                        for (n, l) in zip(self.levels, self.clabels):
                            dic_labels[n] = l
                        if len(self.clabels_mask) > 0:
                            self.ax_tr.clabel(cs, self.levels[self.clabels_mask], fmt=dic_labels,
                                              **self.param_labels_contours)
                        else:
                            self.ax_tr.clabel(cs, self.levels, fmt=dic_labels, **self.param_labels_contours)
                    else:
                        self.ax_tr.clabel(cs, **self.param_labels_contours)
                else:  # axis_config == "br"
                    cs = self.ax_br.contour(self.x_axe_contours, self.y_axe_contours,
                                            self.array_contours, levels, **params)
                    if len(self.clabels) > 0:
                        dic_labels: dict = {}
                        for (n, l) in zip(self.levels, self.clabels):
                            dic_labels[n] = l
                        if len(self.clabels_mask) > 0:
                            self.ax_br.clabel(cs, self.levels[self.clabels_mask], fmt=dic_labels,
                                              **self.param_labels_contours)
                        else:
                            self.ax_br.clabel(cs, self.levels, fmt=dic_labels, **self.param_labels_contours)
                    else:
                        self.ax_br.clabel(cs, **self.param_labels_contours)

            if self.title != "":
                self.ax.set_title(self.title)

    def plot_polygones(self) -> None:
        """
        Plot the Graphique's polygones (if there are)
        Returns
        -------
        None

        """
        with mpl.rc_context(self.param_font):
            for i in range(len(self.index_polygons)):
                params: dict = self.param_polygons[i].copy()
                if "axis_config" in params.keys():
                    axis_config: str = params["axis_config"]
                    del params["axis_config"]
                else:  # bl
                    axis_config: str = "bl"
                P = Path(self.index_polygons[i])
                poly = PathPatch(P, **params)

                if axis_config == "bl":
                    self.ax.add_patch(poly)
                elif axis_config == "tl":
                    self.ax_tl.add_patch(poly)
                elif axis_config == "tr":
                    self.ax_tr.add_patch(poly)
                else:  # axis_config == "br"
                    self.ax_br.add_patch(poly)

    def plot(self, in_Multigraph: bool = False) -> None:
        """
        Plot all the Graphique's elements

        Parameters
        ----------
        in_Multigraph: bool, optional, default=False
            If the Graphique is plotted in a Multigraph, it deactivate globals parameters such as
            the police or the style.

        Returns
        -------
            None

        """
        param_font: dict = {}
        if not in_Multigraph:
            if self.style in plt.style.available or self.style == 'default':
                plt.style.use(self.style)
            else:
                print("The style ", self.style, " is not awalible. \n Plese use :\n", plt.style.available)
            if self.style == 'dark_background':
                self.dark_background()
            elif self.style == 'default':
                self.default_style()
            param_font = self.param_font
        with (mpl.rc_context(param_font)):
            self.param_ax["title"] = self.title
            if self.fig is None:
                self.fig = plt.figure()
            if len(self.param_fig) > 0:
                self.fig.set(**self.param_fig)
            if self.ax is None and "projection" in self.param_ax:
                self.ax = plt.subplot(projection=self.param_ax["projection"])
                if self.axes is None:
                    self.axes = [self.ax]
                else:
                    self.axes.append(self.ax)
            elif self.ax is None:
                self.ax = plt.subplot()
                if self.axes is None:
                    self.axes = [self.ax]
                else:
                    self.axes.append(self.ax)
            elif self.axes is None:
                self.axes = [self.ax]

            if self.custum_colorbar_colors is not None:
                self.plot_colorbar()

            index: np.ndarray[int] = np.arange(0, len(self.lines_x))
            if len(self.indexs_plot_lines) > 0:
                index = np.array(self.indexs_plot_lines)

            if (self.ax_tl is None
                    and (np.any(["axis_config" in dic.keys() and dic["axis_config"] == "tl"
                                 for dic in np.array(self.param_lines)[index]])
                         or "axis_config" in self.param_image and self.param_image["axis_config"] == "tl"
                         or "axis_config" in self.param_contours and self.param_contours["axis_config"] == "tl"
                         or np.any(["axis_config" in dic.keys() and dic["axis_config"] == "tl"
                                    for dic in self.param_polygons]))):
                self.ax_tl = self.ax.twiny()
                self.ax_tl.set(**self.param_ax_tl)
            if (self.ax_br is None
                    and (np.any(["axis_config" in dic.keys() and dic["axis_config"] == "br"
                                 for dic in np.array(self.param_lines)[index]])
                         or "axis_config" in self.param_image and self.param_image["axis_config"] == "br"
                         or "axis_config" in self.param_contours and self.param_contours["axis_config"] == "br"
                         or np.any(["axis_config" in dic.keys() and dic["axis_config"] == "br"
                                    for dic in self.param_polygons]))):
                self.ax_br = self.ax.twinx()
                self.ax_br.set(**self.param_ax_br)
            if (self.ax_tr is None
                and (np.any(["axis_config" in dic.keys() and dic["axis_config"] == "tr"
                            for dic in np.array(self.param_lines)[index]])
                    or "axis_config" in self.param_image and self.param_image["axis_config"] == "tr"
                    or "axis_config" in self.param_contours and self.param_contours["axis_config"] == "tr"
                    or np.any(["axis_config" in dic.keys() and dic["axis_config"] == "tr"
                               for dic in self.param_polygons]))):
                if self.ax_br is None:
                    self.ax_br = self.ax.twinx()
                    self.ax_br.set_visible(False)
                self.ax_tr = self.ax_br.twiny()
                self.ax_tr.set(**self.param_ax_tr)

            if len(self.param_ax) > 0:
                args = self.param_ax.copy()
                if "projection" in args:
                    del args["projection"]
                self.ax.set(**args)

            if len(self.x_axe) == 0 or self.x_axe[0] > -np.inf:
                if len(self.x_axe) > 0:
                    self.ax.set_xlim([self.x_axe.min(), self.x_axe.max()])
                self.ax.set_xticks(self.x_axe)
            if len(self.labels_x_ticks) == 0 or self.labels_x_ticks[0] != "empty":
                self.ax.set_xticklabels(self.labels_x_ticks)
            if len(self.y_axe) == 0 or self.y_axe[0] > -np.inf:
                if len(self.y_axe) > 0:
                    self.ax.set_ylim([self.y_axe.min(), self.y_axe.max()])
                self.ax.set_yticks(self.y_axe)
            if len(self.labels_y_ticks) == 0 or self.labels_y_ticks[0] != "empty":
                self.ax.set_yticklabels(self.labels_y_ticks)
            if self.title != "":
                self.ax.set_title(self.title)

            if len(self.lines_x) > 0:
                self.plot_lines()
            if len(self.vals_histogramme) > 0:
                self.plot_histogrammes()
            if len(self.array_image) > 1:
                self.plot_image()
            if len(self.array_contours) > 1 or (self.tab_contours_is_image and len(self.array_image) > 1):
                self.plot_contours()
            if len(self.index_polygons) > 0:
                self.plot_polygones()
            if len(self.lines_t_x) > 0:
                self.plot_texts()

            if (np.any(["label" in dic.keys() and ("axis_config" not in dic.keys()
                                                   or dic["axis_config"] == "bl")
                        for dic in np.array(self.param_lines)[index]])
                    or np.any(["label" in dic for dic in self.param_histogrammes])
                    or np.any(["label" in dic.keys() and ("axis_config" not in dic.keys()
                                                          or dic["axis_config"] == "bl")
                               for dic in self.param_polygons])):
                self.ax.legend(**self.param_legende)

            if (np.any(["label" in dic.keys() and "axis_config" in dic.keys() and dic["axis_config"] == "tl"
                        for dic in np.array(self.param_lines)[index]])
                    or np.any(["label" in dic for dic in self.param_histogrammes])
                    or np.any(["label" in dic.keys() and "axis_config" in dic.keys() and dic["axis_config"] == "tl"
                               for dic in self.param_polygons])):
                self.ax_tl.legend(**self.param_legende)

            if (np.any(["label" in dic.keys() and "axis_config" in dic.keys() and dic["axis_config"] == "tr"
                        for dic in np.array(self.param_lines)[index]])
                    or np.any(["label" in dic for dic in self.param_histogrammes])
                    or np.any(["label" in dic.keys() and "axis_config" in dic.keys() and dic["axis_config"] == "tr"
                               for dic in self.param_polygons])):
                self.ax_tr.legend(**self.param_legende)

            if (np.any(["label" in dic.keys() and "axis_config" in dic.keys() and dic["axis_config"] == "br"
                        for dic in np.array(self.param_lines)[index]])
                    or np.any(["label" in dic for dic in self.param_histogrammes])
                    or np.any(["label" in dic.keys() and "axis_config" in dic.keys() and dic["axis_config"] == "br"
                               for dic in self.param_polygons])):
                self.ax_br.legend(**self.param_legende)

            self.ax.grid(self.grid)

    def save_figure(self, **args) -> None:
        """
        Save the image product by the Graphique's plotting, not the Graphique itself.
        The image is saved on the Graphiqu's format (default .png)

        Parameters
        ----------
        args
            Additionals parameters for the Figure saving
            see https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html#matplotlib.figure.Figure.savefig

            - figsize 2-tuple of floats, default: rcParams["figure.figsize"] (default: [6.4, 4.8])
                Figure dimension (width, height) in inches.

            - dpi float, default: rcParams["figure.dpi"] (default: 100.0)
                Dots per inch.

            - facecolor default: rcParams["figure.facecolor"] (default: 'white')
                The figure patch facecolor.

            - edgecolor default: rcParams["figure.edgecolor"] (default: 'white')
                The figure patch edge color.

            - linewidth float
                The linewidth of the frame (i.e. the edge linewidth of the figure patch).

            - frameon bool, default: rcParams["figure.frameon"] (default: True)
                If False, suppress drawing the figure background patch.

            - layout {'onstrained', 'compressed', 'tight', 'none', LayoutEngine, None}, default: None

                The layout mechanism for positioning of plot elements to avoid overlapping Axes decorations
                 (labels, ticks, etc). Note that layout managers can have significant performance penalties.

                    'constrained': The constrained layout solver adjusts Axes sizes to avoid overlapping Axes
                     decorations. Can handle complex plot layouts and colorbars, and is thus recommended.

                    See Constrained layout guide for examples.

                    'compressed': uses the same algorithm as 'constrained', but removes extra space between
                    fixed-aspect-ratio Axes. Best for simple grids of Axes.

                    'tight': Use the tight layout mechanism. This is a relatively simple algorithm that adjusts the
                     subplot parameters so that decorations do not overlap.

                    See Tight layout guide for examples.

                    'none': Do not use a layout engine.

                    A LayoutEngine instance. Builtin layout classes are ConstrainedLayoutEngine and TightLayoutEngine,
                     more easily accessible by 'constrained' and 'tight'. Passing an instance allows third parties to
                      provide their own layout engine.

                If not given, fall back to using the parameters tight_layout and constrained_layout, including their
                 config defaults rcParams["figure.autolayout"] (default: False) and
                 rcParams["figure.constrained_layout.use"] (default: False).

            - alpha  scalar or None

            - animated  bool

            - clip_on bool

            - constrained_layout  unknown

            - constrained_layout_pads unknown

            - dpi float

            - edgecolor color

            - facecolor  color

            - figheight float

            - figwidth float

            - frameon bool

            - gid str

            - in_layout bool

            - label object

            - layout_engine {'constrained', 'compressed', 'tight', 'none', LayoutEngine, None}

            - linewidth number

            - mouseover bool

            - picker None or bool or float

            - rasterized bool

            - size_inches (float, float) or float

            - sketch_params (scale: float, length: float, randomness: float)

            - snap bool or None

            - tight_layout unknown

            - transform  Transform

            - url str

            - visible bool

            - zorder float

        Returns
        -------
        None

        """
        args_enrg = self.param_enrg_fig.copy()
        args_enrg.update(args)
        self.plot()
        if "." not in self.ext:
            self.ext = "." + self.ext
        plt.savefig(self.directory + "/" +
                    self.filename + self.ext, **args_enrg)
        self.ax = None
        self.axes = None
        self.ax_tl = None
        self.ax_tr = None
        self.ax_br = None
        plt.close()
        self.fig = None

    def show(self) -> None:
        """
        Show the Graphique

        Returns
        -------
            None

        """
        # plt.ion()
        self.plot()
        plt.show()
        self.ax = None
        self.axes = None
        self.ax_tl = None
        self.ax_tr = None
        self.ax_br = None
        self.fig = None


# ---------------------------------------------------------------------------
# ---------------------------- Graphique building ---------------------------
# ---------------------------------------------------------------------------


def line(x: np.ndarray | list, y: np.ndarray | list | None = None,
         z: np.ndarray | list | None = None,
         marker: str | list = "", share_colorbar: bool = True,
         scale_z: str = "linear", kwargs_colorbar: dict | None = None,
         show: bool = True, hide: bool = False, axis_config: str = "bl", **kwargs) -> Graphique:
    """
    Equivalent to plt.plot

    Parameters
    ----------
    x : array_like | list
        Abscissa(s)
    y : array_like | list, optional
        Ordinate(s), if None x became the ordinate and the abscissa is arange(len(x))
    z : array_like | list, optional
        z-axis (represented by a colorscale)
    marker : str | list[str] | array_like[str], optional, default=""
        The marker  ex ".", ",", "o", "v"... (see matplotlib documentation)
    share_colorbar : bool, optional, default=True
        If True (default) and z is not None, only one colorscale is used
        even if z is in two dimensions
    scale_z : str, {'linear', 'log', 'symlog'}, optional, default='linear'
        The scale of the z-axis
    show : bool, optional, default = True
        To show the Graphique
    hide : bool, optional, default=False
        If True then the new line(s) is/are not plotted with the Graphique.
        To plot them, then change the plot order with self.set_indexs_plot_lines
    kwargs_colorbar : optional
        Extra arguments for the colorbar (if z is not None)
    axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
        The positions of x-y axis :
            - "bl" is x-axis on the bottom, y-axis on the left (default).
            - "tl" is x-axis on the top, y-axis on the left.
            - "br" is x-axis on the bottom, y-axis on the right.
            - "tr" is x-axis on the top, y-axis on the right.

    kwargs, optional
        Additional argument to plot() function like linestyle, color....

    Returns
    -------
    Graphique
        The coresponding new Graphique

    See Also
    --------
    Graphique.loglog : Graphique.line in log coordinate for boths x and y axis
    Graphique.logx : Graphique.line in log coordinate for x axis and linear for y axis
    Graphique.logy : Graphique.line in log coordinate for y axis and linear for x axis
    Graphique.point : To plot a single point
    Graphique.errorbar : To plot a line with errorbars
    Graphique.errorplot : To plot a line with errorbars represanted as filled area
    Graphique.polar : To plot a line in polar coordinates
    Graphique.symloglog : Similar to Graphique.loglog but boths negatives and positives values are represanted
    Graphique.symlogx : Similar to Graphique.logx but boths negatives and positives values are represanted
    Graphique.symlogy : Similar to Graphique.logy but boths negatives and positives values are represanted

    Notes
    -------
    This function has a small improuvment compared with plt.plot :

    if y is in two dimensions, the second dimension is plotted :
        - ```self.line(x,[y1,y2], *args)``` is equivalent to
            ```plt.plot(x, y1, *args)
            plt.plot(x, y2, *args)```
        - if y1 and y2 have not the same size:
            ```line([x1,x2],[y1, y2], *args)```
        - If others arguments are list of the same size of x and y, they are also split :
            ```line((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"])```
            is equivalent to :
            ```plt.plot(x, y1, marker=".", label="Curve1")
            plt.plot(x, y2, marker=".", label="Curve2")```


    Examples
    --------
    >>> x = np.linspace(0, 10, 1000)
    >>> alpha = np.linspace(1, 5, 10)
    >>> colors = linear_color_interpolation(np.arange(len(alpha)), col_min=g.C1, col_max=g.C2)
    >>> gr = line(x, [x*a for a in alpha], color=colors)
    >>> gr.customized_cmap(alpha, colors)
    >>> gr.show()

    """
    if axis_config not in ["bl", "tl", "tr", "br"]:
        raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

    graph: Graphique = Graphique()
    graph.line(x, y, z=z, marker=marker, share_colorbar=share_colorbar,
               scale_z=scale_z, kwargs_colorbar=kwargs_colorbar, hide=hide, axis_config=axis_config, **kwargs)
    if show:
        graph.show()
    return graph


def errorbar(x: list | np.ndarray, y: list | np.ndarray, err_y: list | np.ndarray,
             marker: str = "", scale: str = "", show: bool = True, hide: bool = False,
             axis_config: str = "bl", **kwargs: dict) -> Graphique:
    """
    Equivalent to plt.errorbar

    Parameters
    ----------
    x : list | array_like
        Abscissa
    y : list | array_like
        Ordinate
    err_y : list | array_like
        Error associated with y
    err_x : list | array_like
        Error associated with x
    marker : list[str] | array_like[str], str, optional, default=""
        The marker (ex ".", ",", "o", "v"...)
        see matplotlib documentation for all the possibility
    scale : str, optional, default="linear"
        The scales of (x, y) axis :
        - default : "" (linear scale for both x and y)
        - polar : polar projection : X=R and Y=Theta
        - loglog, logx, logy : Logarithmic scale for both, x or y axis
        - symloglog, symlogx, symlogy : Logarithmic scale for both, x or y axis with positive and négative values

    show : bool, optional, default = True
        To show the Graphique
    hide: bool, optional, default=False
        If True then the new line(s) is/are not plotted with the Graphique.
        To plot them, then change the plot order with self.set_indexs_plot_lines
    axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
        The positions of x-y axis :
            - "bl" is x-axis on the bottom, y-axis on the left (default).
            - "tl" is x-axis on the top, y-axis on the left.
            - "br" is x-axis on the bottom, y-axis on the right.
            - "tr" is x-axis on the top, y-axis on the right.

    kwargs
        Additional argument to plot() function like linestyle, color....

    Returns
    -------
    Graphique
        The new Graphique

    """
    if axis_config not in ["bl", "tl", "tr", "br"]:
        raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

    graph: Graphique = Graphique()
    graph.errorbar(x=x, y=y, err_y=err_y, marker=marker, scale=scale, hide=hide, axis_config=axis_config, **kwargs)
    if show:
        graph.show()
    return graph


def errorplot(x: list | np.ndarray, y: list | np.ndarray, err_y: list | np.ndarray,
              marker: str = "", scale: str = "", show: bool = True, hide: bool = False,
              axis_config: str = "bl", **kwargs: dict) -> Graphique:
    """
    Equivalent to plt.errorbar but the error is not represented by errorbars but by a
    uniform-colored polygon

    Parameters
    ----------
    x : list | array_like
        Abscissa
    y : list | array_like
        Ordinate
    err_y : list | array_like
        Error associated with y
    marker : list[str] | array_like[str], str, optional, default=""
        The marker (ex ".", ",", "o", "v"...)
        see matplotlib documentation for all the possibility
    scale : str, optional, default="linear"
        The scales of (x, y) axis :
        - default : "" (linear scale for both x and y)
        - polar : polar projection : X=R and Y=Theta
        - loglog, logx, logy : Logarithmic scale for both, x or y axis
        - symloglog, symlogx, symlogy : Logarithmic scale for both, x or y axis with positive and négative values

    show : bool, optional, default = True
        To show the Graphique
    hide: bool, optional, default=False
        If True then the new line(s) is/are not plotted with the Graphique.
        To plot them, then change the plot order with self.set_indexs_plot_lines
    kwargs
        Additional argument to plot() function like linestyle, color....
    axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
        The positions of x-y axis :
            - "bl" is x-axis on the bottom, y-axis on the left (default).
            - "tl" is x-axis on the top, y-axis on the left.
            - "br" is x-axis on the bottom, y-axis on the right.
            - "tr" is x-axis on the top, y-axis on the right.


    Returns
    -------
    Graphique
        The new Graphique

    """
    if axis_config not in ["bl", "tl", "tr", "br"]:
        raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

    graph: Graphique = Graphique()
    graph.errorplot(x=x, y=y, err_y=err_y, marker=marker, scale=scale, hide=hide, axis_config=axis_config, **kwargs)
    if show:
        graph.show()
    return graph


def polar(R: list | np.ndarray, Theta: list | np.ndarray,
          z: np.ndarray | list | None = None, marker: str = "",
          share_colorbar: bool = True, scale_z: str = "linear", kwargs_colorbar: dict | None = None,
          axis_config: str = "bl", show: bool = True, hide: bool = False, **kwargs: dict) -> Graphique:
    """
    Equivalent to line in polar projection

    Parameters
    ----------
    R : list | array_like
        Radius
    Theta : list | array_like
        Angle(s)
    z : list | array_like, optional
        z-axis (represented by a colorscale)
    marker:  : list[str] | array_like, str, optional, default=""
        The marker (ex ".", ",", "o", "v"...)
        see matplotlib documentation for all the possibility
    share_colorbar : bool, optional, default=True
        If True(default) and z is not None, only one colorscale is used
        even if z is in two dimensions
    scale_z: str, optional, {"linear", "log", "symlog"}, default="linear"
        The scale of the z-axis (linear (default), log, symplog)
    show : bool, optional, default = True
        To show the Graphique
    hide : bool, optional, default=False
        If True then the new line(s) is/are not plotted with the Graphique.
        To plot them, then change the plot order with self.set_indexs_plot_lines
    kwargs_colorbar: dict
        Extra arguments for the colorbar (if z is not None)
    axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
        The positions of x-y axis :
            - "bl" is x-axis on the bottom, y-axis on the left (default).
            - "tl" is x-axis on the top, y-axis on the left.
            - "br" is x-axis on the bottom, y-axis on the right.
            - "tr" is x-axis on the top, y-axis on the right.

    kwargs
        Additional argument to plot() function like linestyle, color...

    Returns
    -------
    Graphique
        The new Graphique

    Notes
    -------
    The order of first and second arguments is opposit to the matplotlib one :
    The first argument is the radius, then the angle

    """
    if axis_config not in ["bl", "tl", "tr", "br"]:
        raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

    graph: Graphique = Graphique()
    graph.polar(R, Theta, z=z, marker=marker, share_colorbar=share_colorbar,
                scale_z=scale_z, hide=hide, kwargs_colorbar=kwargs_colorbar, axis_config=axis_config, **kwargs)
    if show:
        graph.show()
    return graph


def loglog(x: np.ndarray | list, y: np.ndarray | list | None = None,
           z: np.ndarray | list | None = None,
           marker: str | list = "", share_colorbar: bool = True,
           scale_z: str = "linear", kwargs_colorbar: dict | None = None,
           axis_config: str = "bl", show: bool = True, hide: bool = False, **kwargs) -> Graphique:
    """
    Equivalent to line with a logarithmique scale for both x and y-axis:

    Parameters
    ----------
    x : array_like | list
        Abscissa(s)
    y : array_like | list, optional
        Ordinate(s), if None x became the ordinate and the abscissa is arange(len(x))
    z : array_like | list, optional
        z-axis (represented by a colorscale)
    marker : str | list[str] | array_like[str], optional, default=""
        The marker  ex ".", ",", "o", "v"... (see matplotlib documentation)
    share_colorbar : bool, optional, default=True
        If True (default) and z is not None, only one colorscale is used
        even if z is in two dimensions
    scale_z : str, {'linear', 'log', 'symlog'}, optional, default='linear'
        The scale of the z-axis
    show : bool, optional, default = True
        To show the Graphique
    hide : bool, optional, default=False
        If True then the new line(s) is/are not plotted with the Graphique.
        To plot them, then change the plot order with self.set_indexs_plot_lines
    kwargs_colorbar, optional
        Extra arguments for the colorbar (if z is not None)
    axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
        The positions of x-y axis :
            - "bl" is x-axis on the bottom, y-axis on the left (default).
            - "tl" is x-axis on the top, y-axis on the left.
            - "br" is x-axis on the bottom, y-axis on the right.
            - "tr" is x-axis on the top, y-axis on the right.

    kwargs, optional
        Additional argument to plot() function like linestyle, color....

    Returns
    -------
    Graphique
        The new graphique

    See Also
    --------
    Graphique.line : Build line(s) for the Graphique
    Graphique.logx : Graphique.line in log coordinate for x axis and linear for y axis
    Graphique.logy : Graphique.line in log coordinate for y axis and linear for x axis
    Graphique.point : To plot a single point
    Graphique.errorbar : To plot a line with errorbars
    Graphique.errorplot : To plot a line with errorbars represanted as filled area
    Graphique.polar : To plot a line in polar coordinates
    Graphique.symloglog : Similar to Graphique.loglog but boths negatives and positives values are represanted
    Graphique.symlogx : Similar to Graphique.logx but boths negatives and positives values are represanted
    Graphique.symlogy : Similar to Graphique.logy but boths negatives and positives values are represanted

    Notes
    -------
    This function has a small improuvment compared with plt.plot :

    if y is in two dimensions, the second dimension is plotted :
        - ```loglog(x,[y1,y2], *args)``` is equivalent to
            ```plt.loglog(x, y1, *args)
            plt.loglog(x, y2, *args)```
        - if y1 and y2 have not the same size:
            ```loglog([x1,x2],[y1, y2], *args)```
        - If others arguments are list of the same size of x and y, they are also split :
            ```loglog((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"])```
            is equivalent to
            ```plt.loglog(x, y1, marker=".", label="Curve1")
            plt.loglog(x, y2, marker=".", label="Curve2")```


    Examples
    --------
    >>> x = np.linspace(0, 10, 1000)
    >>> alpha = np.linspace(1, 5, 10)
    >>> colors = linear_color_interpolation(np.arange(len(alpha)), col_min=g.C1, col_max=g.C2)
    >>> gr: Graphique = loglog(x, [x*a for a in alpha], color=colors)
    >>> gr.customized_cmap(alpha, colors)
    >>> gr.show()

    """
    if axis_config not in ["bl", "tl", "tr", "br"]:
        raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

    graph: Graphique = Graphique()
    graph.loglog(x, y, z=z, marker=marker, share_colorbar=share_colorbar,
                 scale_z=scale_z, hide=hide, kwargs_colorbar=kwargs_colorbar, axis_config=axis_config, **kwargs)
    if show:
        graph.show()
    return graph


def symloglog(x: np.ndarray | list, y: np.ndarray | list | None = None,
              z: np.ndarray | list | None = None,
              marker: str | list = "", share_colorbar: bool = True,
              scale_z: str = "linear", kwargs_colorbar: dict | None = None,
              axis_config: str = "bl", show: bool = True, hide: bool = False, **kwargs) -> Graphique:
    """
    Equivalent to line with a logarithmique scale for both x and y-axis
    Both the negative and positive parts of y are represanted:

    Parameters
    ----------
    x : array_like | list
        Abscissa(s)
    y : array_like | list, optional
        Ordinate(s), if None x became the ordinate and the abscissa is arange(len(x))
    z : array_like | list, optional
        z-axis (represented by a colorscale)
    marker : str | list[str] | array_like[str], optional, default=""
        The marker  ex ".", ",", "o", "v"... (see matplotlib documentation)
    share_colorbar : bool, optional, default=True
        If True (default) and z is not None, only one colorscale is used
        even if z is in two dimensions
    scale_z : str, {'linear', 'log', 'symlog'}, optional, default='linear'
        The scale of the z-axis
    show : bool, optional, default = True
        To show the Graphique
    hide : bool, optional, default=False
        If True then the new line(s) is/are not plotted with the Graphique.
        To plot them, then change the plot order with self.set_indexs_plot_lines
    kwargs_colorbar, optional
        Extra arguments for the colorbar (if z is not None)
    axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
        The positions of x-y axis :
            - "bl" is x-axis on the bottom, y-axis on the left (default).
            - "tl" is x-axis on the top, y-axis on the left.
            - "br" is x-axis on the bottom, y-axis on the right.
            - "tr" is x-axis on the top, y-axis on the right.

    kwargs, optional
        Additional argument to plot() function like linestyle, color....

    Returns
    -------
    Graphique
        The new Graphique

    See Also
    --------
    Graphique.line : Build line(s) for the Graphique
    Graphique.loglog : Graphique.line in log coordinate for boths x and y axis
    Graphique.logx : Graphique.line in log coordinate for x axis and linear for y axis
    Graphique.logy : Graphique.line in log coordinate for y axis and linear for x axis
    Graphique.point : To plot a single point
    Graphique.errorbar : To plot a line with errorbars
    Graphique.errorplot : To plot a line with errorbars represanted as filled area
    Graphique.polar : To plot a line in polar coordinates
    Graphique.symloglog : Similar to Graphique.loglog but boths negatives and positives values are represanted
    Graphique.symlogx : Similar to Graphique.logx but boths negatives and positives values are represanted
    Graphique.symlogy : Similar to Graphique.logy but boths negatives and positives values are represanted

    Notes
    -------
    This function has a small improuvment compared with plt.plot :

    if y is in two dimensions, the second dimension is plotted :
        - ```symloglog(x,[y1,y2], *args)``` is equivalent to
            ```ax = plt.subplot()
            ax.plot(x, y1, *args)
            ax.plot(x, y2, *args)
            ax.set(xscale="symlog", yscale="symlog")```
        - if y1 and y2 have not the same size:
            ```symloglog([x1,x2],[y1, y2], *args)```
        - If others arguments are list of the same size of x and y, they are also split :
            ```symloglog((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"])```
            is equivalent to
            ```ax = plt.subplot()
            ax.plot(x, y1, marker=".", label="Curve1")
            ax.plot(x, y2, marker=".", label="Curve2")
            ax.set(xscale="symlog", yscale="symlog")```


    Examples
    --------
    >>> x = np.linspace(-10, 10, 1000)
    >>> gr = symloglog(x, np.tan(x))
    >>> gr.show()

    """
    if axis_config not in ["bl", "tl", "tr", "br"]:
        raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

    graph: Graphique = Graphique()
    graph.symloglog(x, y, z=z, marker=marker, share_colorbar=share_colorbar,
                    scale_z=scale_z, hide=hide, kwargs_colorbar=kwargs_colorbar, axis_config=axis_config, **kwargs)
    if show:
        graph.show()
    return graph


def logx(x: np.ndarray | list, y: np.ndarray | list | None = None,
         z: np.ndarray | list | None = None,
         marker: str | list = "", share_colorbar: bool = True,
         scale_z: str = "linear", kwargs_colorbar: dict | None = None,
         axis_config: str = "bl", show: bool = True, hide: bool = False, **kwargs) -> Graphique:
    """
    Equivalent to line with a logarithmique scale for x-axis:

    Parameters
    ----------
    x : array_like | list
        Abscissa(s)
    y : array_like | list, optional
        Ordinate(s), if None x became the ordinate and the abscissa is arange(len(x))
    z : array_like | list, optional
        z-axis (represented by a colorscale)
    marker : str | list[str] | array_like[str], optional, default=""
        The marker  ex ".", ",", "o", "v"... (see matplotlib documentation)
    share_colorbar : bool, optional, default=True
        If True (default) and z is not None, only one colorscale is used
        even if z is in two dimensions
    scale_z : str, {'linear', 'log', 'symlog'}, optional, default='linear'
        The scale of the z-axis
    show : bool, optional, default = True
        To show the Graphique
    hide : bool, optional, default=False
        If True then the new line(s) is/are not plotted with the Graphique.
        To plot them, then change the plot order with self.set_indexs_plot_lines
    kwargs_colorbar, optional
        Extra arguments for the colorbar (if z is not None)
    axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
        The positions of x-y axis :
            - "bl" is x-axis on the bottom, y-axis on the left (default).
            - "tl" is x-axis on the top, y-axis on the left.
            - "br" is x-axis on the bottom, y-axis on the right.
            - "tr" is x-axis on the top, y-axis on the right.

    kwargs, optional
        Additional argument to plot() function like linestyle, color....

    Returns
    -------
    Graphique
        The new Graphique

    See Also
    --------
    Graphique.line : Build line(s) for the Graphique
    Graphique.loglog : Graphique.line in log coordinate for boths x and y axis
    Graphique.logy : Graphique.line in log coordinate for y axis and linear for x axis
    Graphique.point : To plot a single point
    Graphique.errorbar : To plot a line with errorbars
    Graphique.errorplot : To plot a line with errorbars represanted as filled area
    Graphique.polar : To plot a line in polar coordinates
    Graphique.symloglog : Similar to Graphique.loglog but boths negatives and positives values are represanted
    Graphique.symlogx : Similar to Graphique.logx but boths negatives and positives values are represanted
    Graphique.symlogy : Similar to Graphique.logy but boths negatives and positives values are represanted

    Notes
    -------
    This function has a small improuvment compared with plt.plot :

    if y is in two dimensions, the second dimension is plotted :
        - ```logx(x,[y1,y2], *args)``` is equivalent to
            ```ax = plt.subplot()
            ax.plot(x, y1, *args)
            ax.plot(x, y2, *args)
            ax.set(scale="log")```
        - if y1 and y2 have not the same size:
            ```logx([x1,x2],[y1, y2], *args)```
        - If others arguments are list of the same size of x and y, they are also split :
            ```logx((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"])```
            is equivalent to
            ```ax = plt.subplot()
            ax.plot(x, y1, marker=".", label="Curve1")
            ax.plot(x, y2, marker=".", label="Curve2")
            ax.set(xscale="log")```


    Examples
    --------
    >>> x = np.logspace(-10, 10, 1000)
    >>> gr = Graphique()
    >>> gr.logx(x, np.arctan(x))
    >>> gr.show()

    """
    if axis_config not in ["bl", "tl", "tr", "br"]:
        raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

    graph: Graphique = Graphique()
    graph.logx(x, y, z=z, marker=marker, share_colorbar=share_colorbar,
               scale_z=scale_z, hide=hide, kwargs_colorbar=kwargs_colorbar, axis_config=axis_config, **kwargs)
    if show:
        graph.show()
    return graph


def symlogx(x: np.ndarray | list, y: np.ndarray | list | None = None,
            z: np.ndarray | list | None = None,
            marker: str | list = "", share_colorbar: bool = True,
            scale_z: str = "linear", kwargs_colorbar: dict | None = None,
            axis_config: str = "bl", show: bool = True, hide: bool = False, **kwargs) -> Graphique:
    """
    Equivalent to line with a logarithmique scale for both x-axis (both negative and positive
    part are represanted):

    Parameters
    ----------
    x : array_like | list
        Abscissa(s)
    y : array_like | list, optional
        Ordinate(s), if None x became the ordinate and the abscissa is arange(len(x))
    z : array_like | list, optional
        z-axis (represented by a colorscale)
    marker : str | list[str] | array_like[str], optional, default=""
        The marker  ex ".", ",", "o", "v"... (see matplotlib documentation)
    share_colorbar : bool, optional, default=True
        If True (default) and z is not None, only one colorscale is used
        even if z is in two dimensions
    scale_z : str, {'linear', 'log', 'symlog'}, optional, default='linear'
        The scale of the z-axis
    show : bool, optional, default = True
        To show the Graphique
    hide : bool, optional, default=False
        If True then the new line(s) is/are not plotted with the Graphique.
        To plot them, then change the plot order with self.set_indexs_plot_lines
    kwargs_colorbar, optional
        Extra arguments for the colorbar (if z is not None)
    axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
        The positions of x-y axis :
            - "bl" is x-axis on the bottom, y-axis on the left (default).
            - "tl" is x-axis on the top, y-axis on the left.
            - "br" is x-axis on the bottom, y-axis on the right.
            - "tr" is x-axis on the top, y-axis on the right.

    kwargs, optional
        Additional argument to plot() function like linestyle, color....

    Returns
    -------
    Graphique
        The new Graphique

    See Also
    --------
    Graphique.line : Build line(s) for the Graphique
    Graphique.loglog : Graphique.line in log coordinate for boths x and y axis
    Graphique.logx : Graphique.line in log coordinate for x axis and linear for y axis
    Graphique.logy : Graphique.line in log coordinate for y axis and linear for x axis
    Graphique.point : To plot a single point
    Graphique.errorbar : To plot a line with errorbars
    Graphique.errorplot : To plot a line with errorbars represanted as filled area
    Graphique.polar : To plot a line in polar coordinates
    Graphique.symloglog : Similar to Graphique.loglog but boths negatives and positives values are represanted
    Graphique.symlogy : Similar to Graphique.logy but boths negatives and positives values are represanted

    Notes
    -------
    This function has a small improuvment compared with plt.plot :

    if y is in two dimensions, the second dimension is plotted :
        - ```symlogx(x,[y1,y2], *args)``` is equivalent to
            ```ax = plt.subplot()
            ax.plot(x, y1, *args)
            ax.plot(x, y2, *args)
            ax.set(xscale="symlog")```
        - if y1 and y2 have not the same size:
            ```symlogx([x1,x2],[y1, y2], *args)```
        - If others arguments are list of the same size of x and y, they are also split :
            ```symlogx((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"])```
            is equivalent to
            ```ax = plt.subplot()
            ax.plot(x, y1, marker=".", label="Curve1")
            ax.plot(x, y2, marker=".", label="Curve2")
            ax.set(xscale="symlog")```


    Examples
    --------
    >>> x = np.append(-np.logspace(10, -10, 1000), np.logspace(-10, 10, 1000))
    >>> gr = Graphique()
    >>> gr.symlogx(x, np.arctan(x))
    >>> gr.show()

    """
    if axis_config not in ["bl", "tl", "tr", "br"]:
        raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

    graph: Graphique = Graphique()
    graph.symlogx(x, y, z=z, marker=marker, share_colorbar=share_colorbar,
                  scale_z=scale_z, hide=hide, kwargs_colorbar=kwargs_colorbar, axis_config=axis_config, **kwargs)
    if show:
        graph.show()
    return graph


def logy(x: np.ndarray | list, y: np.ndarray | list | None = None,
         z: np.ndarray | list | None = None,
         marker: str | list = "", share_colorbar: bool = True,
         scale_z: str = "linear", kwargs_colorbar: dict | None = None,
         axis_config: str = "bl", show: bool = True, hide: bool = False, **kwargs) -> Graphique:
    """

    Equivalent to line with a logarithmique scale for y-axis:

    Parameters
    ----------
    x : array_like | list
        Abscissa(s)
    y : array_like | list, optional
        Ordinate(s), if None x became the ordinate and the abscissa is arange(len(x))
    z : array_like | list, optional
        z-axis (represented by a colorscale)
    marker : str | list[str] | array_like[str], optional, default=""
        The marker  ex ".", ",", "o", "v"... (see matplotlib documentation)
    share_colorbar : bool, optional, default=True
        If True (default) and z is not None, only one colorscale is used
        even if z is in two dimensions
    scale_z : str, {'linear', 'log', 'symlog'}, optional, default='linear'
        The scale of the z-axis
    show : bool, optional, default = True
        To show the Graphique
    hide : bool, optional, default=False
        If True then the new line(s) is/are not plotted with the Graphique.
        To plot them, then change the plot order with self.set_indexs_plot_lines
    kwargs_colorbar, optional
        Extra arguments for the colorbar (if z is not None)
    axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
        The positions of x-y axis :
            - "bl" is x-axis on the bottom, y-axis on the left (default).
            - "tl" is x-axis on the top, y-axis on the left.
            - "br" is x-axis on the bottom, y-axis on the right.
            - "tr" is x-axis on the top, y-axis on the right.

    kwargs, optional
        Additional argument to plot() function like linestyle, color....

    Returns
    -------
    Graphique
        The new Graphique

    See Also
    --------
    Graphique.line : Build line(s) for the Graphique
    Graphique.loglog : Graphique.line in log coordinate for boths x and y axis
    Graphique.logx : Graphique.line in log coordinate for x axis and linear for y axis
    Graphique.point : To plot a single point
    Graphique.errorbar : To plot a line with errorbars
    Graphique.errorplot : To plot a line with errorbars represanted as filled area
    Graphique.polar : To plot a line in polar coordinates
    Graphique.symloglog : Similar to Graphique.loglog but boths negatives and positives values are represanted
    Graphique.symlogx : Similar to Graphique.logx but boths negatives and positives values are represanted
    Graphique.symlogy : Similar to Graphique.logy but boths negatives and positives values are represanted

    Notes
    -------
    This function has a small improuvment compared with plt.plot :

    if y is in two dimensions, the second dimension is plotted :
        - ```logy(x,[y1,y2], *args)``` is equivalent to
            ```ax = plt.subplot()
            ax.plot(x, y1, *args)
            ax.plot(x, y2, *args)
            ax.set(yscale="log")```
        - if y1 and y2 have not the same size:
            ```logy([x1,x2],[y1, y2], *args)```
        - If others arguments are list of the same size of x and y, they are also split :
            ```logy((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"]```
            is equivalent to
            ```ax = plt.subplot()
            ax.plot(x, y1, marker=".", label="Curve1")
            ax.plot(x, y2, marker=".", label="Curve2")
            ax.set(yscale="log")```


    Examples
    --------
    >>> x = np.logspace(-10, 10, 1000)
    >>> gr = logy(x, np.arctan(x))
    >>> gr.show()

    """
    if axis_config not in ["bl", "tl", "tr", "br"]:
        raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

    graph: Graphique = Graphique()
    graph.logy(x, y, z=z, marker=marker, share_colorbar=share_colorbar,
               scale_z=scale_z, hide=hide, kwargs_colorbar=kwargs_colorbar, axis_config=axis_config, **kwargs)
    if show:
        graph.show()
    return graph


def symlogy(x: np.ndarray | list, y: np.ndarray | list | None = None,
            z: np.ndarray | list | None = None,
            marker: str | list = "", share_colorbar: bool = True,
            scale_z: str = "linear", kwargs_colorbar: dict | None = None,
            axis_config: str = "bl", show: bool = True, hide: bool = False, **kwargs) -> Graphique:
    """
    Equivalent to line with a logarithmique scale for y-axis (both positive and negative
    part are represanted):

    Parameters
    ----------
    x : array_like | list
        Abscissa(s)
    y : array_like | list, optional
        Ordinate(s), if None x became the ordinate and the abscissa is arange(len(x))
    z : array_like | list, optional
         z-axis (represented by a colorscale)
    marker : str | list[str] | array_like[str], optional, default=""
        The marker  ex ".", ",", "o", "v"... (see matplotlib documentation)
    share_colorbar : bool, optional, default=True
         If True (default) and z is not None, only one colorscale is used
         even if z is in two dimensions
    scale_z : str, {'linear', 'log', 'symlog'}, optional, default='linear'
        The scale of the z-axis
    show : bool, optional, default = True
        To show the Graphique
    hide : bool, optional, default=False
        If True then the new line(s) is/are not plotted with the Graphique.
        To plot them, then change the plot order with self.set_indexs_plot_lines
    kwargs_colorbar, optional
        Extra arguments for the colorbar (if z is not None)
    axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
        The positions of x-y axis :
            - "bl" is x-axis on the bottom, y-axis on the left (default).
            - "tl" is x-axis on the top, y-axis on the left.
            - "br" is x-axis on the bottom, y-axis on the right.
            - "tr" is x-axis on the top, y-axis on the right.

    kwargs, optional
        Additional argument to plot() function like linestyle, color....

    Returns
    -------
    Graphique
        The new Graphique

    See Also
    --------
    Graphique.line : Build line(s) for the Graphique
    Graphique.loglog : Graphique.line in log coordinate for boths x and y axis
    Graphique.logy : Graphique.line in log coordinate for y axis and linear for x axis
    Graphique.point : To plot a single point
    Graphique.errorbar : To plot a line with errorbars
    Graphique.errorplot : To plot a line with errorbars represanted as filled area
    Graphique.polar : To plot a line in polar coordinates
    Graphique.symloglog : Similar to Graphique.loglog but boths negatives and positives values are represanted
    Graphique.symlogx : Similar to Graphique.logx but boths negatives and positives values are represanted

    Notes
    -------
    This function has a small improuvment compared with plt.plot :

    if y is in two dimensions, the second dimension is plotted :
        - ```symlogy(x,[y1,y2], *args)``` is equivalent to
            ```ax = plt.subplot()
            ax.plot(x, y1, *args)
            ax.plot(x, y2, *args)
            ax.set(yscale="symlog")```
        - if y1 and y2 have not the same size:
            ```symlogy([x1,x2],[y1, y2], *args)```
        - If others arguments are list of the same size of x and y, they are also split :
            ```symlogy((x1, x2], [y1, y2], marker=".", label=["Curve1", "Curve2"])```
            is equivalent to
            ```ax = plt.subplot()
            ax.plot(x, y1, marker=".", label="Curve1")
            ax.plot(x, y2, marker=".", label="Curve2")
            ax.set(yscale="symlog")```


    Examples
    --------
    >>> x = np.linspace(0,np.pi,1000)
    >>> gr = gr.symlogy(x, np.tan(x))
    >>> gr.show()

    """
    if axis_config not in ["bl", "tl", "tr", "br"]:
        raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

    graph: Graphique = Graphique()
    graph.symlogy(x, y, z=z, marker=marker, share_colorbar=share_colorbar,
                  scale_z=scale_z, hide=hide, kwargs_colorbar=kwargs_colorbar, axis_config=axis_config, **kwargs)
    if show:
        graph.show()
    return graph


def histogram(values: np.ndarray, weights: np.ndarray | None = None,
              normalization: bool = True, statistic: str = 'sum', bins: int = 10,
              show: bool = True, **args) -> Graphique:
    """
    Plot the histogram of values

    Parameters
    ----------
    values : array_like
        The values to histogramed
    weights : array_like, optional
        The weights to be applied to values
    normalization : bool, optional, default=True
        If the histogram is normalized or not
    statistic : str, optional, default="sum"
        The statistic to compute (default is 'sum').
        The following statistics are available:

            - 'mean': compute the mean of values for points within each bin. Empty bins will be represented by NaN.
            - 'std': compute the standard deviation within each bin. This is implicitly calculated with ddof=0.
            - 'median': compute the median of values for points within each bin. Empty bins will be represented by NaN.
            - 'count': compute the count of points within each bin. This is identical to an unweighted histogram. values array is not
                referenced.
            - 'sum': compute the sum of values for points within each bin. This is identical to a weighted histogram.
            - 'min': compute the minimum of values for points within each bin. Empty bins will be represented by NaN.
            - 'max': compute the maximum of values for point within each bin. Empty bins will be represented by NaN.

    bins : int, optional, default=10
        Number of bins in the histogram
    stat_args, dict, optional
        Additionals argument for `sp.binned_statistic`
    show : bool, optional, default = True
        To show the Graphique
    kwargs
         Additionals argument for plt.bars

    Returns
    -------
    Graphique
        The new Graphique

    See also
    --------
    scipy.stats.binned_statistic

    """
    if weights is None:
        poids = []
    graph: Graphique = Graphique()
    graph.histogram(
        values=values, weights=weights, normalization=normalization, statistic=statistic,
        bins=bins, **args)
    if show:
        graph.show()
    return graph


def image(array_image: np.ndarray,
          x_axe: list | np.ndarray, y_axe: list | np.ndarray,
          axis_config: str = "bl", show: bool = True, **args) -> Graphique:
    """

    Plot the array image through plt.pcolor or plt.imshow for 3 color images

    Parameters
    ----------
    array_image : np.ndarray
        The matrix (2D, or 3D for colored images (the color is on the third axis)) to be plotted
    x_axe : list | np.ndarray, optional, default=np.linespace(0,array_image.shape[0])
        The x-axes coordinate (for the array), only for 2d array_image
    y_axe : list | np.ndarray, optional, default=np.linespace(0,array_image.shape[1])
        The y-axes coordinate (for the array), only for 2d array_image
    colorscale : str, optional, default="linear", {"linear", "log", "symlog"}
        The scale for the colorbar
    cmap : str, optional, default="default"
        The colormap, default a linear color interpolation between two colors CX
    colorbar_ticks : list | array_like, optional
        The colorbar's ticks
    colorbar_label : str, optional, default=""
        The colorbar's label
    kwargs_colorbar : dict, optional
        Additional arguments for the colorbar :
            - location: str, {'right', 'top', 'bottom', 'left'}
                Indicate where the colorbar should be plotted
            - scale: str, {'linear', 'log', 'symlog'}
                The scale of the colorbar
            - ticks: list | array_like
            - format: str
                ticks' format
            - label: str
                The label to plot along the colorbar
            - size: float, default=0.01
                relative width of the colorbar
            - fraction: float, default=1
                relative hight of the colorbar
            - space_between: float, default=0.01
                relative space between colorsbars (and the plot)

    colorbar_index : int, optional
        The index of a previouly defined colorbar, to use insted of building a new one, all the others colorbar are
        ignored if provided
    vmin : np.float64, optional, default : the minimum of array_image
        The minimum value for the colorbar
    vmax : np.float64, optional, default : the maximum of array_image
        The maximum value for the colorbar
    axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
        The positions of x-y axis :
            - "bl" is x-axis on the bottom, y-axis on the left (default).
            - "tl" is x-axis on the top, y-axis on the left.
            - "br" is x-axis on the bottom, y-axis on the right.
            - "tr" is x-axis on the top, y-axis on the right.

    show : bool, optional, default = True
        To show the Graphique
    kwargs
        Additionals arguments for pcolor

    Returns
    -------
        None

    See Also
    --------

    matplotlib.pyplot.pcolor
        Use to plot 2d array images
    matplotlib.pyplot.imshow
        Use to plot 3-colors images
    Graphique.contours
        To draw levels lines onto the image

    """
    if axis_config not in ["bl", "tl", "tr", "br"]:
        raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

    graph: Graphique = Graphique()
    graph.image(array_image=array_image, x_axe=x_axe, y_axe=y_axe, axis_config=axis_config, **args)
    if show:
        graph.show()
    return graph


def level_surface(x: np.ndarray | list, y: np.ndarray | list,
                  vals: np.ndarray | list, npix_x: int = 400, npix_y: int = 400,
                  logx: bool = False, logy: bool = False,
                  method: str = 'cubic', log_vals: bool = False, axis_config: str = "bl",
                  show: bool = True, **kwargs) -> Graphique:
    """

    Returns an image representing the 2d contour line associated with the points
    points defined by x, y, vals

    Parameters
    ----------
    x : np.ndarray | list
        list of size n and dimension 1
        containing the abscissa of the points
    y : np.ndarray | list
        list of size n and dimension 1
        containing the odrides of the points
    vals : np.ndarray | list
        list of size n and dimension 1
        containing point values to be interpolated and displayed
    npix_x : int, optional, default = 400
        number of image pixels on x axis
    npix_y : int, optional, default = 400
        number of image pixels on y axis
    logx : bool, optional, default = False
        indicates whether the x axis is logarithmically subdivided or not
    logy : bool, optional, default = False
        indicates whether the y axis is logarithmically subdivided or not
    method : str, optional, default = "cubic
        interplotation method: 'nearest', 'linear' where by default
        'cubic'. See doc scipy.interpolate.griddata
    log_vals : bool, optional, default = False
        indicates whether values are displayed (and extrapolated) on a logarithmic scale or not.
        on a logarithmic scale
    axis_config : str, optional, {"bl", "tl", "br", "tr"}, default="bl"
        The positions of x-y axis :

            - "bl" is x-axis on the bottom, y-axis on the left (default).
            - "tl" is x-axis on the top, y-axis on the left.
            - "br" is x-axis on the bottom, y-axis on the right.
            - "tr" is x-axis on the top, y-axis on the right.

    show : bool, optional, default = True
        To show the Graphique (default=True)
    kwargs
        dictionaries of complementary arguments to images
        
    Returns
    -------
    Graphique
        The new Graphique

    """
    if axis_config not in ["bl", "tl", "tr", "br"]:
        raise UserWarning("""The axis configuration can only be "bl", "tl"; "tr" or "br", not """, axis_config)

    points: np.ndarray = np.array([x, y]).T
    if logx:
        x_int: np.ndarray = np.geomspace(np.min(x), np.max(x), npix_x)
    else:
        x_int: np.ndarray = np.linspace(np.min(x), np.max(x), npix_x)
    if logy:
        y_int: np.ndarray = np.geomspace(np.min(y), np.max(y), npix_y)
    else:
        y_int: np.ndarray = np.linspace(np.min(y), np.max(y), npix_y)
    xi_x, xi_y = np.meshgrid(x_int, y_int)
    if log_vals:
        tab: np.ndarray = griddata(points, np.log10(vals),
                                   xi=(xi_x, xi_y), method=method, rescale=True)
        tab = 10 ** tab
        print(tab.min(), tab.max())
    else:
        tab: np.ndarray = griddata(points, vals, xi=(xi_x, xi_y), method=method)

    res: Graphique = image(tab, x_int, y_int, shading='nearest', show=False, axis_config=axis_config,
                           **kwargs)
    res.config_colorbar(scale="log")
    if show:
        res.show()
    return res


class Multigraph:
    """
    To plot multiples Graphiques in a single Figure

    """

    def __init__(self, nrows: int = 1, ncols: int = 1, filename: str = "", directory: str = ""):
        """
        To build a Multigraph

        Parameters
        ----------
        nrows : int, optional, default=1
            Numbers of rows in the array containing the Graphiques
        ncols : int, optional, default=1
            Numbers of collumns in the array containing the Graphiques
        filename : str, optyional, default=""
            To load an existing Multigraph (if not "")
        directory : str, optyional, default=""
            The directory of the loaded Multigraph, if filename != "". If filename
            alreaday contain the total path, the directory is automaticaly deduced

        """
        if '.npz' in filename:
            filename = filename[:-4]

        if filename == "":
            self.directory: str = "./"
            self.filename: str = "graph_without_name"
        self.ext: str = ".png"
        self.style: str = 'default'  # global style:
        # styles available: plt.style.available : 'default' 'Solarize_Light2'
        # '_classic_test_patch' '_mpl-gallery' '_mpl-gallery-nogrid'
        # 'bmh' 'classic' 'dark_background' 'fast' 'fivethirtyeight' 'ggplot'
        # 'grayscale' 'seaborn' 'seaborn-bright' 'seaborn-colorblind' 'seaborn-dark' 'seaborn-dark-palette'
        # 'seaborn-darkgrid' 'seaborn-deep' 'seaborn-muted' 'seaborn-notebook' 'seaborn-paper' 'seaborn-pastel'
        # 'seaborn-poster' 'seaborn-talk' 'seaborn-ticks' 'seaborn-white' 'seaborn-whitegrid' 'tableau-colorblind10'
        self.param_font: dict = {"font.size": 13}  # Contains additional parameters for global font management

        self.param_fig: dict = dict()  # Contains additional parameters for the Figure ex : facecolor="w"...
        self.param_grid_spec: dict = dict()  # Contains additional parameters for the grid : hspace, vspace...
        self.param_enrg_fig: dict = dict(bbox_inches="tight")

        self.nrows: int = nrows
        self.ncols: int = ncols
        self.list_Graphs: list[Graphique] = []
        self.list_coords: list[int | list[int | tuple]] = []
        self.list_sharex_axis_coord: list[int | tuple] = []
        self.list_sharey_axis_coord: list[int | tuple] = []
        self.param_subplot: list[dict] = []
        self.occupy_coordinates: list[list[int]] = []  # List of coordinates of the occupy slots to test if a Graphique
        # could be added with a new set of coordinate
        self.fig = None
        if filename != "":
            self.filename = filename
            if directory != "":
                self.directory = directory
            elif "/" in directory:
                i: int = directory.find("/")
                while directory.find("/") > 0:
                    # Cherche la dernière occurrence de "/"
                    i = directory.find("/")
                self.directory = filename[:i]
                self.filename = filename[i:]
            else:
                self.directory = "./"
            values_files: np.lib.npyio.NpzFile = np.load(directory + filename + ".npz")
            values_to_load: dict = ndarray_dict_to_dict(dict(values_files))
            values_files.close()
            self.load_dict(values_to_load)

    def load_dict(self, values_to_load: dict) -> None:
        """

        load a Graphix contain in a dictionary

        Parameters
        ----------
        values_to_load : dict
            Dictionary containing all the information needed to build a Multigraph.
            This dictionary is produced by Multigraph.to_dict()

        Returns
        -------
        None

        See Also
        --------
        Multigraph.to_dict
            Produce a dictionary that can be used to build a new Multigraph

        """
        if "ext" in values_to_load.keys():
            self.ext: str = values_to_load["ext"]
        if "style" in values_to_load.keys():
            self.style: str = values_to_load["style"]
        if "param_font" in values_to_load.keys():
            self.param_font: str = values_to_load["param_font"]
        if "param_fig" in values_to_load.keys():
            self.param_fig: str = values_to_load["param_fig"]
        if "param_enrg_fig" in values_to_load.keys():
            self.param_enrg_fig: str = values_to_load["param_enrg_fig"]
        if "param_grid_spec" in values_to_load.keys():
            self.param_grid_spec: str = values_to_load["param_grid_spec"]
        if "nrows" in values_to_load.keys():
            self.nrows: str = values_to_load["nrows"]
        if "ncols" in values_to_load.keys():
            self.ncols: str = values_to_load["ncols"]
        if "list_Graphs" in values_to_load.keys():
            self.list_Graphs: str = values_to_load["list_Graphs"]
        if "list_coords" in values_to_load.keys():
            self.list_coords: str = values_to_load["list_coords"]
        if "list_sharex_axis_coord" in values_to_load.keys():
            self.list_sharex_axis_coord: str = values_to_load["list_sharex_axis_coord"]
        if "list_sharey_axis_coord" in values_to_load.keys():
            self.list_sharey_axis_coord: str = values_to_load["list_sharey_axis_coord"]
        if "occupy_coordinates" in values_to_load.keys():
            self.occupy_coordinates: str = values_to_load["occupy_coordinates"]
        if "param_subplot" in values_to_load.keys():
            self.param_subplot: str = values_to_load["param_subplot"]

    def to_dict(self) -> dict:
        """

        Build a dictionary to be saved with np.saved_compressed

        Returns
        -------
        dict

        See Also
        --------

        Multigraph.load_dict
            To load the produced dictionary in another Multigraph

        """
        enrg: dict = dict()  # Dictionary containing all the necessary information :
        # Used like :  np.savez_compressed(name_fichier,**enrg)

        enrg["ext"] = self.ext
        enrg["style"] = self.style
        enrg["param_font"] = self.param_font
        enrg["param_fig"] = self.param_fig
        enrg["param_enrg_fig"] = self.param_enrg_fig
        enrg["param_grid_spec"] = self.param_grid_spec
        enrg["nrows"] = self.nrows
        enrg["ncols"] = self.ncols
        enrg["list_Graphs"] = self.list_Graphs
        enrg["list_coords"] = self.list_coords
        enrg["list_sharex_axis_coord"] = self.list_sharex_axis_coord
        enrg["list_sharey_axis_coord"] = self.list_sharey_axis_coord
        enrg["occupy_coordinates"] = self.occupy_coordinates
        enrg["param_subplot"] = self.param_subplot

        return enrg

    def save(self, filename: str = "mgraph_without_name", directory: str = None) -> None:
        """
        Save the Multi.graph in self.directory (default the current working directory) in npz compressed
        format.

        Parameters
        ----------
        filename: str, optinal, default="mgraph_without_name"
            The name of the .npz file (default: "mgraph_without_name")
        directory: str, optional, default="./"
            Graphique's directory (default self.directory (default : the curent working directory))

        Returns
        -------
        None

        """
        if filename != "mgraph_without_name":
            if ".npz" in filename:
                self.filename = filename[:-4]
            else:
                self.filename = filename
        if directory is not None:
            self.directory = directory
        enrg = dict_to_ndarray_dict(self.to_dict(), separator="-.-")
        if ".npz" not in self.filename:
            if self.directory[-1] == "/":
                np.savez_compressed(self.directory +
                                    self.filename + ".npz", **enrg)
            else:
                np.savez_compressed(self.directory + "/" +
                                    self.filename + ".npz", **enrg)
        else:
            if self.directory[-1] == "/":
                np.savez_compressed(self.directory +
                                    self.filename, **enrg)
            else:
                np.savez_compressed(self.directory + "/" +
                                    self.filename, **enrg)

    def add_graphique(self, coord: int | list[int, tuple], graph: Graphique = None,
                      sharex: int | list[int, tuple] = None,
                      sharey: int | list[int, tuple] = None, **kwargs):
        """
        To add a new Graphique in the Multigraph at the coordinates defined by coord.
        This Graphique can cover several slots

        Parameters
        ----------
        coord : int | list[int, tuple]
            the Graphique's coordinates : if self.nrows or nself.ncols =1 it can be an int, either a list
            of int :
            [index_rows, index_cols]. If this Graphique extend on severals slots, coord should be on this form :
            [(index_rows_min, index_rows_max), (index_cols_min, index_cols_min)]
        graph : Graphique, optional, default : an empty Graphique
            The Graphique to add, default a new empty Graphique is created
        sharex : int | list[int, tuple], optional
            Coordinates of the Graphique to wich the x axis is shared if needed. The corresponding Graphique
            should already be in the Multigraph. If the new Graphique is on multiples slots then the Graphique to which the
            coordinates are shared should occupy the same number of raw.
        sharey : int | list[int, tuple], optional
            Coordinates of the Graphique to wich the y axis is shared if needed. The corresponding Graphique
            should already be in the Multigraph. If the new Graphique is on multiples slots then the Graphique to which the
            coordinates are shared should occupy the same number of cols
        kwargs
            Extra-parameters for plt.subplot

        Returns
        -------
        None

        """
        if graph is None:
            graph = Graphique()
        elif not isinstance(graph, Graphique):
            raise UserWarning("Multigraph.add_Graphique : graph should be a Graphique not a ", type(graph))
        if isinstance(coord, int) and self.nrows > 1 and self.ncols > 1:
            raise UserWarning("Multigraph.add_Graphique : The Multigraph isn't linear nrow=",
                              self.nrows, "ncols=", self.ncols,
                              "The coordinate of the added Graphique should be a list, not an int")
        elif isinstance(coord, int) and coord >= self.nrows * self.ncols:
            raise UserWarning("Multigraph.add_Graphique : The coordinate of the added Graphique ",
                              coord, "is higher than the maximum :", self.nrows * self.ncols - 1)
        elif isinstance(coord, list) and isinstance(coord[0], int) and coord[0] >= self.nrows:
            raise UserWarning("Multigraph.add_Graphique : The first coordinate of the added Graphique ",
                              coord[0], "is higher than the maximum :", self.nrows - 1)
        elif isinstance(coord, list) and isinstance(coord[1], int) and coord[1] >= self.ncols:
            raise UserWarning("Multigraph.add_Graphique : The second coordinate of the added Graphique ",
                              coord[1], "is higher than the maximum :", self.nrows - 1)
        elif isinstance(coord, list) and isinstance(coord[0], tuple) and coord[0][1] >= self.nrows:
            raise UserWarning("Multigraph.add_Graphique : The first maximum coordinate of the added Graphique ",
                              coord[0][1], "is higher than the maximum :", self.nrows - 1)
        elif isinstance(coord, list) and isinstance(coord[1], tuple) and coord[1][1] >= self.ncols:
            raise UserWarning("Multigraph.add_Graphique : The second maximum coordinate of the added Graphique ",
                              coord[1][1], "is higher han the maximum :", self.ncols - 1)
        occupy_coordinates: list = []
        if isinstance(coord, int):
            if self.ncols == 1:
                if coord < 0:
                    coord = (self.nrows + coord) % self.nrows
                if [coord, 0] in self.occupy_coordinates or [coord, 0] in occupy_coordinates:
                    raise UserWarning("Multigraph.add_graphique : "
                                      "The coordinate ", [coord, 0], "is already occupy")
                else:
                    occupy_coordinates.append([coord, 0])
                coord: list = [coord, 0]
            else:
                if coord < 0:
                    coord = (self.cols + coord) % self.ncols
                if [0, coord] in self.occupy_coordinates or [0, coord] in occupy_coordinates:
                    raise UserWarning("Multigraph.add_graphique : The coordinate ", [0, coord], "is already occupy")
                else:
                    occupy_coordinates.append([0, coord])
                coord: list = [0, coord]
        elif isinstance(coord, list) and isinstance(coord[0], int) and isinstance(coord[1], int):
            if coord in self.occupy_coordinates or coord in occupy_coordinates:
                raise UserWarning("Multigraph.add_graphique : The coordinate ", coord, "is already occupy")
            else:
                occupy_coordinates.append(coord)
        elif isinstance(coord, list) and isinstance(coord[0], int):
            for i in range(coord[1][0], coord[1][1] + 1):
                if [coord[0], i] in self.occupy_coordinates or [coord[0], i] in occupy_coordinates:
                    raise UserWarning("Multigraph.add_graphique : The coordinate ", [coord[0], i], "is already occupy")
                else:
                    occupy_coordinates.append([coord[0], i])
        elif isinstance(coord, list) and isinstance(coord[1], int):
            for i in range(coord[0][0], coord[0][1] + 1):
                if [i, coord[1]] in self.occupy_coordinates or [i, coord[1]] in occupy_coordinates:
                    raise UserWarning("Multigraph.add_graphique : The coordinate ", [i, coord[1]], "is already occupy")
                else:
                    occupy_coordinates.append([i, coord[1]])
        else:
            for i in range(coord[0][0], coord[0][1] + 1):
                for j in range(coord[1][0], coord[1][1] + 1):
                    if [i, j] in self.occupy_coordinates or [i, j] in occupy_coordinates:
                        raise UserWarning("Multigraph.add_graphique : The coordinate ", [i, j], "is already occupy")
                    else:
                        occupy_coordinates.append([i, j])
        if sharex is not None and sharex not in self.list_coords:
            raise UserWarning("Multigraph.add_graphique : there is no Graphique at the coordinates indicates"
                              " for the sharex parameter")
        elif sharex is not None:
            if coord[1] != sharex[1]:
                raise UserWarning(
                    "Multigraph.add_graphique : the second coordinates of the sharex Graphique doesn't match "
                    "with the new coordinates")
        else:
            sharex = [ii_max, ii_max]

        if sharey is not None and sharey not in self.list_coords:
            raise UserWarning("Multigraph.add_graphique : there is no Graphique at the coordinates indicates"
                              " for the sharey parameter")
        elif sharey is not None:
            if coord[0] != sharey[0]:
                raise UserWarning(
                    "Multigraph.add_graphique : the first coordinates of the sharey Graphique doesn't match "
                    "with the new coordinates")
        else:
            sharey = [ii_max, ii_max]
        self.occupy_coordinates.extend(occupy_coordinates)
        self.list_Graphs.append(graph)
        self.list_coords.append(coord)
        self.list_sharex_axis_coord.append(sharex)
        self.list_sharey_axis_coord.append(sharey)
        self.param_subplot.append(kwargs)

    def config_fig(self, **kwargs) -> None:
        """

        Additionals parameters for the Figure saving

        Parameters
        ----------
        kwargs

            - igsize 2-tuple of floats, default: rcParams["figure.figsize"] (default: [6.4, 4.8])
                Figure dimension (width, height) in inches.

            - dpi float, default: rcParams["figure.dpi"] (default: 100.0)
                Dots per inch.

            - facecolor default: rcParams["figure.facecolor"] (default: 'white')
                The figure patch facecolor.

            - edgecolor default: rcParams["figure.edgecolor"] (default: 'white')
                The figure patch edge color.

            - linewidth float
                The linewidth of the frame (i.e. the edge linewidth of the figure patch).

            - frameon bool, default: rcParams["figure.frameon"] (default: True)
                If False, suppress drawing the figure background patch.

            - layout {'onstrained', 'compressed', 'tight', 'none', LayoutEngine, None}, default: None

                The layout mechanism for positioning of plot elements to avoid overlapping Axes decorations
                 (labels, ticks, etc). Note that layout managers can have significant performance penalties.

                    'constrained': The constrained layout solver adjusts Axes sizes to avoid overlapping Axes
                     decorations. Can handle complex plot layouts and colorbars, and is thus recommended.

                    See Constrained layout guide for examples.

                    'compressed': uses the same algorithm as 'constrained', but removes extra space between
                    fixed-aspect-ratio Axes. Best for simple grids of Axes.

                    'tight': Use the tight layout mechanism. This is a relatively simple algorithm that adjusts the
                     subplot parameters so that decorations do not overlap.

                    See Tight layout guide for examples.

                    'none': Do not use a layout engine.

                    A LayoutEngine instance. Builtin layout classes are ConstrainedLayoutEngine and TightLayoutEngine,
                     more easily accessible by 'constrained' and 'tight'. Passing an instance allows third parties to
                      provide their own layout engine.

                If not given, fall back to using the parameters tight_layout and constrained_layout, including their
                 config defaults rcParams["figure.autolayout"] (default: False) and
                 rcParams["figure.constrained_layout.use"] (default: False).

            - alpha  scalar or None

            - animated  bool

            - clip_on bool

            - constrained_layout  unknown

            - constrained_layout_pads unknown

            - dpi float

            - edgecolor color

            - facecolor  color

            - figheight float

            - figwidth float

            - frameon bool

            - gid str

            - in_layout bool

            - label object

            - layout_engine {'constrained', 'compressed', 'tight', 'none', LayoutEngine, None}

            - linewidth number

            - mouseover bool

            - picker None or bool or float

            - rasterized bool

            - size_inches (float, float) or float

            - sketch_params (scale: float, length: float, randomness: float)

            - snap bool or None

            - tight_layout unknown

            - transform  Transform

            - url str

            - visible bool

            - zorder float

        Returns
        -------
        None

        See Also
        --------

        matplotlib.figure.Figure.savefig

        """
        self.param_fig.update(kwargs)

    def config_enrg_fig(self, **kwargs) -> None:
        """

        Additionals parameters for the Figure saving
        see https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html#matplotlib.figure.Figure.savefig

        Parameters
        ----------
        kwargs
            - figsize 2-tuple of floats, default: rcParams["figure.figsize"] (default: [6.4, 4.8])
                Figure dimension (width, height) in inches.

            - dpi float, default: rcParams["figure.dpi"] (default: 100.0)
                Dots per inch.

            - facecolor default: rcParams["figure.facecolor"] (default: 'white')
                The figure patch facecolor.

            - edgecolor default: rcParams["figure.edgecolor"] (default: 'white')
                The figure patch edge color.

            - linewidth float
                The linewidth of the frame (i.e. the edge linewidth of the figure patch).

            - frameon bool, default: rcParams["figure.frameon"] (default: True)
                If False, suppress drawing the figure background patch.

            - layout {'onstrained', 'compressed', 'tight', 'none', LayoutEngine, None}, default: None

                The layout mechanism for positioning of plot elements to avoid overlapping Axes decorations
                 (labels, ticks, etc). Note that layout managers can have significant performance penalties.

                    -- 'constrained': The constrained layout solver adjusts Axes sizes to avoid overlapping Axes
                     decorations. Can handle complex plot layouts and colorbars, and is thus recommended.

                    See Constrained layout guide for examples.

                    -- 'compressed': uses the same algorithm as 'constrained', but removes extra space between
                    fixed-aspect-ratio Axes. Best for simple grids of Axes.

                    -- 'tight': Use the tight layout mechanism. This is a relatively simple algorithm that adjusts the
                     subplot parameters so that decorations do not overlap.

                    See Tight layout guide for examples.

                    'none': Do not use a layout engine.

                    A LayoutEngine instance. Builtin layout classes are ConstrainedLayoutEngine and TightLayoutEngine,
                     more easily accessible by 'constrained' and 'tight'. Passing an instance allows third parties to
                      provide their own layout engine.

                If not given, fall back to using the parameters tight_layout and constrained_layout, including their
                 config defaults rcParams["figure.autolayout"] (default: False) and
                 rcParams["figure.constrained_layout.use"] (default: False).

            - alpha  scalar or None

            - animated  bool

            - clip_on bool

            - constrained_layout  unknown

            - constrained_layout_pads unknown

            - dpi float

            - edgecolor color

            - facecolor  color

            - figheight float

            - figwidth float

            - frameon bool

            - gid str

            - in_layout bool

            - label object

            - layout_engine {'constrained', 'compressed', 'tight', 'none', LayoutEngine, None}

            - linewidth number

            - mouseover bool

            - picker None or bool or float

            - rasterized bool

            - size_inches (float, float) or float

            - sketch_params (scale: float, length: float, randomness: float)

            - snap bool or None

            - tight_layout unknown

            - transform  Transform

            - url str

            - visible bool

            - zorder float

        Returns
        -------
        None

        """
        self.param_enrg_fig.update(kwargs)

    def config_grid_spec(self, **dico) -> None:
        """

        Additionals parameters for the grid_spec

        Parameters
        ----------
        dico

            - left, right, top, bottomfloat, optional : Extent of the subplots as a fraction of figure width or height.
                Left cannot be larger than right, and bottom cannot be larger than top.
                If not given, the values will be inferred from a figure or rcParams at draw time.
                See also GridSpec.get_subplot_params.

            - wspace float, optional :
                The amount of width reserved for space between subplots, expressed as a fraction
                of the average axis width. If not given, the values will be inferred from a figure or rcParams
                when necessary. See also GridSpec.get_subplot_params.

            - hspace float, optional :
                The amount of height reserved for space between subplots, expressed as a
                fraction of the average axis height. If not given, the values will be inferred from a figure or rcParams
                when necessary. See also GridSpec.get_subplot_params.

            - width_ratios array-like of length ncols, optional :
                Defines the relative widths of the columns.
                Each column gets a relative width of width_ratios[i] / sum(width_ratios).
                If not given, all columns will have the same width.

            - height_ratios array-like of length nrows, optional
                Defines the relative heights of the rows.
                Each row gets a relative height of height_ratios[i] / sum(height_ratios).
                If not given, all rows will have the same height.

        Returns
        -------
        None

        See Also
        --------
        matplotlib.gridspec.GridSpec

        """
        self.param_grid_spec.update(dico)

    def config_font(self, **dico) -> None:
        """
        Global font parameter

        Parameters
        ----------
        kwargs
           'family' : 'fantasy','monospace','sans-serif','serif','cursive'
           'styles' : 'normal', 'italic', 'oblique'
           'size' : valeur numérique
           'variants' : 'normal', 'small-caps'
           'weight' : 'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'

        Returns
        -------
        None

        """
        k: list[str] = dico.keys()
        vals: list = dico.values()
        dico: dict = {}
        for K, L in zip(k, vals):
            if "font." not in K:
                dico['font.' + K] = L
            else:
                dico[K] = L
        self.param_font.update(dico)

    def plot(self) -> None:
        """
        Plot all the Multigraph's elements

        Parameters
        ----------
        in_Multigraph: bool, optional, default=False
            If the Graphique is plotted in a Multigraph, it deactivate globals parameters such as
            the police or the style.

        Returns
        -------
            None

        """
        if self.style in plt.style.available or self.style == 'default':
            plt.style.use(self.style)
        else:
            print("The style ", self.style, " is not awalible. \n Please use :\n", plt.style.available)

        with mpl.rc_context(self.param_font):
            param_fig = {'figsize': [6.4 * self.ncols, 4.8 * self.nrows]}
            if len(self.param_fig) > 0:
                param_fig.update(self.param_fig)
            self.fig = plt.figure(**param_fig)

            spec = self.fig.add_gridspec(self.nrows, self.ncols, **self.param_grid_spec)
            for (graph, coords, sharex, sharey, args) in zip(self.list_Graphs, self.list_coords,
                                                             self.list_sharex_axis_coord, self.list_sharey_axis_coord,
                                                             self.param_subplot):
                if isinstance(sharex[0], int) and sharex[0] == ii_max:
                    sharex = None
                elif isinstance(sharex, list):
                    sharex = self.list_Graphs[self.list_coords.index(sharex)].ax
                # else: sharex is bool

                if isinstance(sharey[0], int) and sharey[0] == ii_max:
                    sharey = None
                elif isinstance(sharey, list):
                    sharey = self.list_Graphs[self.list_coords.index(sharey)].ax

                projection: str = None
                graph.fig = self.fig
                if "projection" in graph.param_ax:
                    projection = graph.param_ax["projection"]
                if isinstance(coords[0], int) and isinstance(coords[1], int):
                    # graph.ax = self.fig.add_subplot(spec[coords[0], coords[1]], projection=projection, sharex=sharex,
                    #                            sharey=sharey, **args)
                    graph.ax = self.fig.add_subplot(spec[coords[0], coords[1]], projection=projection, **args)
                    if sharex is not None:
                        graph.ax.sharex(sharex)
                    if sharey is not None:
                        graph.ax.sharey(sharey)
                elif isinstance(coords[0], int):
                    # graph.ax = self.fig.add_subplot(spec[coords[0], coords[1][0]:coords[1][1] + 1], projection=projection,
                    #                            sharex=sharex, sharey=sharey, **args)
                    graph.ax = self.fig.add_subplot(spec[coords[0], coords[1][0]:coords[1][1] + 1],
                                                    projection=projection,
                                                    **args)
                    if sharex is not None:
                        graph.ax.sharex(sharex)
                    if sharey is not None:
                        graph.ax.sharey(sharey)
                elif isinstance(coords[1], int):
                    # graph.ax = self.fig.add_subplot(spec[coords[0][0]:coords[0][1] + 1, coords[1]], projection=projection,
                    #                            sharex=sharex, sharey=sharey, **args)
                    #
                    graph.ax = self.fig.add_subplot(spec[coords[0][0]:coords[0][1] + 1, coords[1]],
                                                    projection=projection,
                                                    **args)
                    if sharex is not None:
                        graph.ax.sharex(sharex)
                    if sharey is not None:
                        graph.ax.sharey(sharey)
                else:
                    # graph.ax = self.fig.add_subplot(spec[coords[0][0]:coords[0][1] + 1, coords[1][0]:coords[1][1] + 1],
                    #                            projection=projection, sharex=sharex, sharey=sharey, **args)
                    graph.ax = self.fig.add_subplot(spec[coords[0][0]:coords[0][1] + 1, coords[1][0]:coords[1][1] + 1],
                                                    projection=projection, **args)
                    if sharex is not None:
                        graph.ax.sharex(sharex)
                    if sharey is not None:
                        graph.ax.sharey(sharey)
                graph.plot(in_Multigraph=True)
        self.fig.tight_layout()

    def save_figure(self, **args) -> None:
        """

        Save the image product by the Multigraph's plotting, not the Multigraph itself.
        The image is saved on the Multigraph's format (default .png)

        Parameters
        ----------
        args
            Additionals parameters for the Figure saving
            see https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html#matplotlib.figure.Figure.savefig

            - igsize 2-tuple of floats, default: rcParams["figure.figsize"] (default: [6.4, 4.8])
                Figure dimension (width, height) in inches.

            - dpi float, default: rcParams["figure.dpi"] (default: 100.0)
                Dots per inch.

            - facecolor default: rcParams["figure.facecolor"] (default: 'white')
                The figure patch facecolor.

            - edgecolor default: rcParams["figure.edgecolor"] (default: 'white')
                The figure patch edge color.

            - linewidth float
                The linewidth of the frame (i.e. the edge linewidth of the figure patch).

            - frameon bool, default: rcParams["figure.frameon"] (default: True)
                If False, suppress drawing the figure background patch.

            - layout {'onstrained', 'compressed', 'tight', 'none', LayoutEngine, None}, default: None

                The layout mechanism for positioning of plot elements to avoid overlapping Axes decorations
                 (labels, ticks, etc). Note that layout managers can have significant performance penalties.

                    'constrained': The constrained layout solver adjusts Axes sizes to avoid overlapping Axes
                     decorations. Can handle complex plot layouts and colorbars, and is thus recommended.

                    See Constrained layout guide for examples.

                    'compressed': uses the same algorithm as 'constrained', but removes extra space between
                    fixed-aspect-ratio Axes. Best for simple grids of Axes.

                    'tight': Use the tight layout mechanism. This is a relatively simple algorithm that adjusts the
                     subplot parameters so that decorations do not overlap.

                    See Tight layout guide for examples.

                    'none': Do not use a layout engine.

                    A LayoutEngine instance. Builtin layout classes are ConstrainedLayoutEngine and TightLayoutEngine,
                     more easily accessible by 'constrained' and 'tight'. Passing an instance allows third parties to
                      provide their own layout engine.

                If not given, fall back to using the parameters tight_layout and constrained_layout, including their
                 config defaults rcParams["figure.autolayout"] (default: False) and
                 rcParams["figure.constrained_layout.use"] (default: False).

            - alpha  scalar or None

            - animated  bool

            - clip_on bool

            - constrained_layout  unknown

            - constrained_layout_pads unknown

            - dpi float

            - edgecolor color

            - facecolor  color

            - figheight float

            - figwidth float

            - frameon bool

            - gid str

            - in_layout bool

            - label object

            - layout_engine {'constrained', 'compressed', 'tight', 'none', LayoutEngine, None}

            - linewidth number

            - mouseover bool

            - picker None or bool or float

            - rasterized bool

            - size_inches (float, float) or float

            - sketch_params (scale: float, length: float, randomness: float)

            - snap bool or None

            - tight_layout unknown

            - transform  Transform

            - url str

            - visible bool

            - zorder float

        Returns
        -------
        None

        """
        args_enrg = self.param_enrg_fig.copy()
        args_enrg.update(args)
        self.plot()
        if "." not in self.ext:
            self.ext = "." + self.ext
        plt.savefig(self.directory + "/" +
                    self.filename + self.ext, **args_enrg)
        self.ax = None
        for graph in self.list_Graphs:
            graph.fig = None
            graph.ax = None
        plt.close()
        self.fig = None

    def show(self) -> None:
        """
        Show the Graphique

        Returns
        -------
            None

        """

        self.plot()
        self.fig.tight_layout()
        plt.show()
        self.fig = None
        for graph in self.list_Graphs:
            graph.fig = None
            graph.ax = None
