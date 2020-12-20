# **Adva Mobileye project**
## Detection traffic lights and the distance to them on runtime within given video

### **phase I:**
Detection of source lights in an image using convolution with
customized high- and low-pass filters.

### **phase II:**
Generating and training CNN using the products of the
previous stage as input, to conclude all the traffic lights in the
image (using tensorflow).

### **phase III:** 
Estimating the distance to each detected traffic light from the
camera picturing the images of interest, involving geometric and
linear algebra calculations.

### **phase IV:** 
Integrating all previous parts into a functional and intuitive SW
product

## Libraries/Technologies Used:
* python 3.7
* numpy
* matplotlib
* scipy
* imgaug
* jupyterlab
* tensorflow
