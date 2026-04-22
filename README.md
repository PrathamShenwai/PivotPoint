# PivotPoint
Airfoil Angle of Attack Tracker

# PivotPoint: Airfoil AoA & LED Tracker

PivotPoint is a computer vision tool designed to track the **Angle of Attack (AoA)** of an airfoil and monitor a signaling **LED** in experimental video footage. It uses OpenCV’s CSRT algorithm for high-accuracy point tracking and logs results to a CSV file.

##  Prerequisites 

1. **Python 3.8+**
2. **Required Libraries:**
   '''pip install opencv-contrib-python numpy pygame'''


## ROI Selection Guide

The accuracy of the data depends entirely on the Region of Interest (ROI) selection. When the program opens, follow these steps precisely:

1. Airfoil ROI: Draw a large box around the ENTIRE airfoil and its full range of expected motion. Press ENTER.

2. LED ROI: Draw a TIGHT box focusing specifically on the black/dark region of the LED. Selecting the dark center allows the software to detect the maximum contrast "spike" when the LED glows red. Press ENTER.

3. Pivot ROI: Draw a box around only the inner circle of the central bolt/pivot. This prevents the tracker from drifting to the outer edges of the hardware. Press ENTER.

4. Leading Edge ROI: Draw a very tight box around the black leading-edge dot. Keeping this selection tight ensures the software doesn't lose the dot against the background. Press ENTER.
