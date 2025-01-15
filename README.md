# PheniGraph

![Logo phenigraph](logo.png)
## Description
Allows you to save a matplotlib plot in a dedicated object so that you can reuse it as is later.

## Badges

## Visuals

## Installation

To compile the library :

In the main folder, run the command :

`python setup.py bdist_wheel`

The installation wheel file is in the newly created ‘dist’ folder.
To install the library in the local environment:

`pip install /path/to/wheelfile.whl`

Then the library can be used like any other:
`import graphique as g`

## Usage
 
This library has two dedicated objects : one for the signe plots : Graphique and one 
for building multiples Graphique in a single figure : Multigraph

### The Graphique object :
#### To initialise a Graphique :

- For a new blank graph: `g=Graphique()`.

- To add content directly to the graph (and display it immediately):
	- `graph=g.line(x,y,marker=‘’,**args)` equivalent to `plt.plot(x,y,marker=‘’,**args);plt.show()` but you can recover the graph
	- `graph=g.errorbar(x,y, err_y, marker=‘’,**args)` equivalent to `plt.errorbar(x,y, err_y, marker=‘’,**args);plt.show()` but the graph can be retrieved.

- To open a graph with the name `n` in the `f` directory
(`n` and `f` are two strings, by default `f` is the current directory)
eg:
	`f=‘../Images/Graphique’` and n=`‘test_graphique’`.
(the .npz extension is not mandatory) :
	- `g=Graphique(n)` if n is in the current directory
	- `g=Graphique(n,f)` otherwise

#### To save a Graphique :
- Assign a name to the Graphique, without using an extension:
	`gr.filename=‘new_name’`
	The default name is graph_without_name.
    If you want to save several graphs in the same folder, it is important to give them a name (not automatic).


- Possibly assign a directory for saving:
	`gr.directory=‘new_directory’` By default, the location is the current directory.

- To save the object:
	`gr.save()`
- To save the figure :
	- Assign an extension if necessary (by default the extension is svg). 
  Possible extensions are those available via the matplotlib library: ‘png’, ‘pdf’, ‘svg’, etc.
	```
	gr.ext=’.new_extension
	gr.save_figure()
	```

#### To display the Graphique :
```gr.show()```

#### To add a line of points (equivalent to plt.plot):

``gr.line(x,y,**args)```
where `x` and `y` are two lists or arrays of data to be displayed and
`**args` corresponds to all the other possible arguments to plot()

Can be repeated as many times as necessary

#### To add a histogram :

`g.histogram(values,weight=[],normalisation=True,statistic=‘sum’, bins=10, range=None,**args)` 

where :
- `values` is the array, the list of values to be classified
- `weight` is a list giving a possible weight for each value
- `normalisation` indicates whether or not the histogram should be normalised
- The other arguments are the same as `plt.hist()`.
Can be repeated as many times as necessary
	

#### To add an image:

`g.image(array,x-axis,y-axis,**args)`
where :
- `array` represents the image to be displayed
- `axis_x` and `axis_y` give the graduations of the image axes
- `**args` all the other possible arguments for displaying an image



#### To add contours:
`g.contours(self,contours=np.array([[]]),axe_x=None,axe_y=None,**args) `
- `**args` gives the possible arguments for plt.contours()
- To add level lines to an image, complete `**args`, leaving the other arguments as defaults



#### To add a polygon (coloured area delimited by a list of points): 

`g.polygon(ind,alpha=0.7, facecolor=‘C3’,**args)`
with `ind` an array/list of dimension (n,2) where `n` is the number of points: `ind[:,0]` corresponds to the abscissae of the points and `ind[:1]` to the ordinates.
Can be repeated as many times as necessary

### The Multigraph :

A Multigraph has a table of shape (nrows, ncols). A Graphique could be assign to one
or several slots of this table.
The usage is very similar than for Graphique.

#### To initialise a Multigraph :

- For a new blank Multigraph: `mg=g.Multigraph(nrows, ncols)`.


- To open a graph with the filename `f` in the `d` directory
(`f` and `d` are two strings, by default `d` is the current directory)
eg:
	`d=‘../Images/Multigraph’` and f=`‘test_multigraph’`.
(the .npz extension is not mandatory) :
	- `mg=g.Multigraph(filename=f)` if n is in the current directory
	- `mg=g.Multigraph(filename=f,diretory=d)` otherwise

#### To save a Multigraph :
- Assign a name to the Multigraph, without using an extension:
	`mg.filename=‘new_name’`
	The default name is graph_without_name.
    If you want to save several graphs in the same folder, it is important to give them a name (not automatic).


- Possibly assign a directory for saving:
	`mg.directory=‘new_directory’` By default, the location is the current directory.

- To save the object:
	`mg.save()`
- To save the figure :
	- Assign an extension if necessary (by default the extension is png). 
  Possible extensions are those available via the matplotlib library: ‘png’, ‘pdf’, ‘svg’, etc.
	```
	mg.ext=’.new_extension'
	mg.save_figure()
	```

#### To display the Multigraph :
```mg.show()```



## Support

## Roadmap

## Contributing

## Authors and acknowledgment
Paul Huet

## License


## Project status
