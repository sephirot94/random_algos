### About project
I made some decisions and assumptions that i would like to clarify briefly:
1. Usage of CV2 library: based on the question “what does it mean to be checked?” I thought about "What makes a checkbox?" in the first place. A checkbox is a square with defined contours, that can be filled or not. A filled checkbox is a checkbox with high pixel density. Hence, i decided to use CV2 library to check contours and pixel density to define what a checked box is.
2. Usage of Makefile: this tool allows for building and running quickly. It's true that we would most likely be using Docker and containers, even kubernetes and pods, to develop and release code to production, not a makefile with a venv and requirements file. However, for the sake of speed and reducing complexity in the design of this challenge, i decided to go with this alternative and leave Docker out of the picture.
3. Not freezing dependency versions (Pipfile): the reasoning behind this is similar to the previous point, but if this is a small project to be handed in a matter of days, the risk of a new dependency version being released and breaking the solution is very low and can be ignored.
4. Not using supervised machine learning: In an ideal world with a lot of data i could have created a model that detects a check box, both empty or checked, with some labeling and supervised training. However this would require a lot of data i do not possess for training. Fun fact, this project could be used for automatic labelling of data used for training.

### How to execute
Open a terminal with the working directory being the root folder of this project and execute the following command:
`make build && make run`
Doing so should execute the program and output a new file with the detected check boxes. Green is used for filled check box while red is for unfilled checkbox. Both filled and unfilled are detected.
To change input parameters, you will need to add a new image to the /images folder and change in main.py:
`input_image_path = "images/sample-section-mod.webp"`
Change it to
`input_image_path = "your/new/image/name"`