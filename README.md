<h1>NSL_ComputerVisionStuff</h1>
Repository for computer vision code for USF NSL 21-22 challenge.<br>
Folder Code/ contains all the python and jupyter files used for testing and development.<br>
Folder MainCode/ contiains the essential python files that were used in the final version that was running on a NVIDIA Jetson Computer that was inside the rocket launched for the Nasa Student Launch 2021-2022 season<br>
The Data/ folder contains the image data used for testing the algorithms

<h2>Program Description </h2>
The code inside sherlock.py contains the main program that's called once the Jetson computer has stored all the images of the rocket descent in memory (refer to /Data folder for some examples on how those would look like).
<br>
Sherlock.py relies on the eyeinthesky.py which holds the core algorithm behind the project. We use image matching between the images the rocket captured on the descent and actual satellite footage of our launching zone, by comparing the two we are able to obtain an estaimate of where our rocket was going to land. The comparison is done via an ML algorithm LOFTR that we are able to use with the help of the Kornia library and OpenCV. Even though LOFTR isn't very accurate a great heights, what we do is we split the satellite imagery into 4 quadrants and make LOFTR iterate over each quadrant until it finds the best matching one. For the algorithm to run as fast as possible we use the NVIDIA Jetson's GPU, that's why we use CUDA in the program.
The final estimate processed by Sherlock is compared with the functions inside thegrid.py to output which is the most probable quadrant we landed on.

<h2> Setup the code </h2>
To access the code setup more easily try forking the Gradient Notebook which uses a cloud GPU to avoid crashing your own computer: 
https://console.paperspace.com/tek5fvbsq/notebook/rikn565w9orvmqp?file=Tutorial%20Docs%2FHow_To_Use.ipynb
