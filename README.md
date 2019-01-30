#FARSI CAR NUMBER PLATE DETECTION AND RECOGNITION

What is a Number Plate?
-----------------------
a number plate is an image which the density of it's vertical lines is high.

Feature Extraction Procedure
===================
* convert rgb image to **grayscale**.
    * it makes detection procedure independent of colors.
* reduce possible noise of image.
    * **gaussian** or bilateral filters used.
* bold vertical edges of image by y-direction derivation.
    * **sobel** filter used.

...
