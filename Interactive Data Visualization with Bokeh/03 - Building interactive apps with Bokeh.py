'''
1. Using the current document
Let's get started with building an interactive Bokeh app. This typically begins with importing the curdoc, or "current document", function from bokeh.io. This current document will eventually hold all the plots, controls, and layouts that you create. Your job in this exercise is to use this function to add a single plot to your application.

In the video, Bryan described the process for running a Bokeh app using the bokeh serve command line tool. In this chapter and the one that follows, the DataCamp environment does this for you behind the scenes. Notice that your code is part of a script.py file. When you hit 'Submit Answer', you'll see in the IPython Shell that we call bokeh serve script.py for you.

Remember, as in the previous chapters, that there are different options available for you to interact with your plots, and as before, you may have to scroll down to view the lower portion of the plots.

INSTRUCTIONS
100 XP
Import curdoc from bokeh.io and figure from bokeh.plotting.
Create a new plot called plot using the figure() function.
Add a line to the plot using [1,2,3,4,5] as the x coordinates and [2,5,4,6,7] as the y coordinates.
Add the plot to the current document using curdoc().add_root(). It needs to be passed in as an argument to add_root().
'''

# Perform necessary imports
from bokeh.io import curdoc
from bokeh.plotting import figure

# Create a new plot: plot
plot = figure()

# Add a line to the plot
plot.line([1,2,3,4,5], [2,5,4,6,7])

# Add the plot to the current document
curdoc().add_root(plot)

'''
2. Add a single slider
In the previous exercise, you added a single plot to the "current document" of your application. In this exercise, you'll practice adding a layout to your current document.

Your job here is to create a single slider, use it to create a widgetbox layout, and then add this layout to the current document.

The slider you create here cannot be used for much, but in the later exercises, you'll use it to update your plots!

INSTRUCTIONS
100 XP
Import curdoc from bokeh.io, widgetbox from bokeh.layouts, and Slider from bokeh.models.
Create a slider called slider by using the Slider() function and specifying the parameters title, start, end, step, and value.
Use the slider to create a widgetbox layout called layout.
Add the layout to the current document using curdoc().add_root(). It needs to be passed in as an argument to add_root().
'''

# Perform the necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# Create a slider: slider
slider = Slider(title='my slider', start=0, end=10, step=0.1, value=2)

# Create a widgetbox layout: layout
layout = widgetbox(slider)

# Add the layout to the current document
curdoc().add_root(layout)

'''
3. Multiple sliders in one document
Having added a single slider in a widgetbox layout to your current document, you'll now add multiple sliders into the current document.

Your job in this exercise is to create two sliders, add them to a widgetbox layout, and then add the layout into the current document.

INSTRUCTIONS
100 XP
Create the first slider, slider1, using the Slider() function. Give it a title of 'slider1'. Have it start at 0, end at 10, with a step of 0.1 and initial value of 2.
Create the second slider, slider2, using the Slider() function. Give it a title of 'slider2'. Have it start at 10, end at 100, with a step of 1 and initial value of 20.
Use slider1 and slider2 to create a widgetbox layout called layout.
Add the layout to the current document using curdoc().add_root(). This has already been done for you.
'''

# Perform necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# Create first slider: slider1
slider1 = Slider(title='slider1', start=0, end=10, step=0.1,value=2)

# Create second slider: slider2
slider2 = Slider(title='slider2', start=10, end=100, step=1, value=20)

# Add slider1 and slider2 to a widgetbox
layout = widgetbox(slider1, slider2)

# Add the layout to the current document
curdoc().add_root(layout)

'''
4. How to combine Bokeh models into layouts
Let's begin making a Bokeh application that has a simple slider and plot, that also updates the plot based on the slider.

In this exercise, your job is to first explicitly create a ColumnDataSource. You'll then combine a plot and a slider into a single column layout, and add it to the current document.

After you are done, notice how in the figure you generate, the slider will not actually update the plot, because a widget callback has not been defined. You'll learn how to update the plot using widget callbacks in the next exercise.

All the necessary modules have been imported for you. The plot is available in the workspace as plot, and the slider is available as slider.

INSTRUCTIONS
100 XP
Create a ColumnDataSource called source. Explicitly specify the data parameter of ColumnDataSource() with {'x': x, 'y': y}.
Add a line to the figure plot, with 'x' and 'y' from the ColumnDataSource.
Combine the slider and the plot into a column layout called layout. Be sure to first create a widgetbox layout using widgetbox() with slider and pass that into the column() function along with plot.
'''

# Create ColumnDataSource: source
source = ColumnDataSource(data={'x': x, 'y': y})

# Add a line to the plot
plot.line('x', 'y', source=source)

# Create a column layout: layout
layout = column(widgetbox(slider), plot)

# Add the layout to the current document
curdoc().add_root(layout)

'''
5 Learn about widget callbacks
You'll now learn how to use widget callbacks to update the state of a Bokeh application, and in turn, the data that is presented to the user.

Your job in this exercise is to use the slider's on_change() function to update the plot's data from the previous example. NumPy's sin() function will be used to update the y-axis data of the plot.

Now that you have added a widget callback, notice how as you move the slider of your app, the figure also updates!

INSTRUCTIONS
100 XP
Define a callback function callback with the parameters attr, old, new.
Read the current value of slider as a variable scale. You can do this using slider.value.
Compute values for the updated y using np.sin(scale/x).
Update source.data with the new data dictionary.
Attach the callback to the 'value' property of slider. This can be done using on_change() and passing in 'value' and callback.
'''

# Define a callback function: callback
def callback(attr, old, new):

    # Read the current value of the slider: scale
    scale = slider.value

    # Compute the updated y using np.sin(scale/x): new_y
    new_y = np.sin(scale/x)

    # Update source with the new data values
    source.data = {'x': x, 'y': new_y}

# Attach the callback to the 'value' property of slider
slider.on_change('value', callback)

# Create layout and add to current document
layout = column(widgetbox(slider), plot)
curdoc().add_root(layout)

'''
6. Updating data sources from dropdown callbacks
You'll now learn to update the plot's data using a drop down menu instead of a slider. This would allow users to do things like select between different data sources to view.

The ColumnDataSource source has been created for you along with the plot. Your job in this exercise is to add a drop down menu to update the plot's data.

All necessary modules have been imported for you.

INSTRUCTIONS
100 XP
Define a callback function called update_plot with the parameters attr, old, new.
If the new selection is female_literacy, update the y value of the ColumnDataSource to female_literacy. Else, y should be population.
x remains fertility in both cases.
Create a dropdown select widget using Select(). Specify the parameters title, options, and value. The options are 'female_literacy' and 'population', while the value is 'female_literacy'.
Attach the callback to the 'value' property of select. This can be done using on_change() and passing in 'value' and update_plot.
'''

# Perform necessary imports
from bokeh.models import ColumnDataSource, Select

# Create ColumnDataSource: source
source = ColumnDataSource(data={
    'x' : fertility,
    'y' : female_literacy
})

# Create a new plot: plot
plot = figure()

# Add circles to the plot
plot.circle('x', 'y', source=source)

# Define a callback function: update_plot
def update_plot(attr, old, new):
    # If the new Selection is 'female_literacy', update 'y' to female_literacy
    if new == 'female_literacy': 
        source.data = {
            'x' : fertility,
            'y' : female_literacy
        }
    # Else, update 'y' to population
    else:
        source.data = {
            'x' : fertility,
            'y' : population
        }

# Create a dropdown Select widget: select    
select = Select(title="distribution", options=['female_literacy', 'population'], value='female_literacy')

# Attach the update_plot callback to the 'value' property of select
select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(select, plot)
curdoc().add_root(layout)

'''
7. Synchronize two dropdowns
Here, you'll practice using a dropdown callback to update another dropdown's options. This will allow you to customize your applications even further and is a powerful addition to your toolbox.

Your job in this exercise is to create two dropdown select widgets and then define a callback such that one dropdown is used to update the other dropdown.

All modules necessary have been imported.

INSTRUCTIONS
100 XP
Create select1, the first dropdown select widget. Specify the parameters title, options, and value.
Create select2, the second dropdown select widget. Specify the parameters title, options, and value.
Inside the callback function, if select1 equals 'A', update the options of select2 to ['1', '2', '3'] and set its value to '1'.
If select1 does not equal 'A', update the options of select2 to ['100', '200', '300'] and set its value to '100'.
Attach the callback to the 'value' property of select1. This can be done using on_change() and passing in 'value' and callback.
'''

# Create two dropdown Select widgets: select1, select2
select1 = Select(title='First', options=['A', 'B'], value='A')
select2 = Select(title='Second', options=['1', '2', '3'], value='1')

# Define a callback function: callback
def callback(attr, old, new):
    # If select1 is 'A' 
    if select1.value == 'A':
        # Set select2 options to ['1', '2', '3']
        select2.options = ['1', '2', '3']

        # Set select2 value to '1'
        select2.value = '1'
    else:
        # Set select2 options to ['100', '200', '300']
        select2.options = ['100', '200', '300']

        # Set select2 value to '100'
        select2.value = '100'

# Attach the callback to the 'value' property of select1
select1.on_change('value', callback)

# Create layout and add to current document
layout = widgetbox(select1, select2)
curdoc().add_root(layout)

'''
8. Button widgets
It's time to practice adding buttons to your interactive visualizations. Your job in this exercise is to create a button and use its on_click() method to update a plot.

All necessary modules have been imported for you. In addition, the ColumnDataSource with data x and y as well as the figure have been created for you and are available in the workspace as source and plot.

When you're done, be sure to interact with the button you just added to your plot, and notice how it updates the data!

INSTRUCTIONS
100 XP
Create a button called button using the function Button() with the label 'Update Data'.
Define an update callback update() with no arguments.
Compute new y values using the code provided.
Update the ColumnDataSource data dictionary source.data with the new y value.
Add the update callback to the button using on_click().
'''

# Create a Button with label 'Update Data'
button = Button(label='Update Data')

# Define an update callback with no arguments: update
def update():

    # Compute new y values: y
    y = np.sin(x) + np.random.random(N)

    # Update the ColumnDataSource data dictionary
    source.data = {'x': x, 'y': y}

# Add the update callback to the button
button.on_click(update)

# Create layout and add to current document
layout = column(widgetbox(button), plot)
curdoc().add_root(layout)

'''
9. Button styles
You can also get really creative with your Button widgets.

In this exercise, you'll practice using CheckboxGroup, RadioGroup, and Toggle to add multiple Button widgets with different styles.

curdoc and widgetbox have already been imported for you.

INSTRUCTIONS
100 XP
Import CheckboxGroup, RadioGroup, Toggle from bokeh.models.
Add a Toggle called toggle using the Toggle() function with button_type 'success' and label 'Toggle button'.
Add a CheckboxGroup called checkbox using the CheckboxGroup() function with labels=['Option 1', 'Option 2', 'Option 3'].
Add a RadioGroup called radio using the RadioGroup() function with labels=['Option 1', 'Option 2', 'Option 3'].
Add a widgetbox containing the Toggle toggle, CheckboxGroup checkbox, and RadioGroup radio to the current document.
'''

# Import CheckboxGroup, RadioGroup, Toggle from bokeh.models
from bokeh.models import CheckboxGroup, RadioGroup, Toggle

# Add a Toggle: toggle
toggle = Toggle(label='Toggle button',button_type='success')

# Add a CheckboxGroup: checkbox
checkbox = CheckboxGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add a RadioGroup: radio
radio = RadioGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add widgetbox(toggle, checkbox, radio) to the current document
curdoc().add_root(widgetbox(toggle, checkbox, radio))
