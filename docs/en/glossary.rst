Glossary
=========

:link_to_translation:`zh_CN:[中文]`

.. glossary::

   Tensor

     A tensor is a generalization of matrices to N dimensions. In other words, it could be:

     -  0-dimensional, represented as a scalar
     -  1-dimensional, represented as a vector
     -  2-dimensional, represented as a matrix
     -  a higher-dimensional structure that is harder to visualize

     The number of dimensions and the size of each dimension is known as the shape of a tensor. In ESP-DL, a tensor is the primary data structure. Every input and output of a layer is a tensor.

     In 2D operations, the input tensor and output tensor of a layer is 3D. Tensor dimensions are ordered in a fixed manner, namely [height, width, channel].

     Suppose we have the following shape [5, 3, 4], and the elements of this tensor would be arranged as follows:

     .. figure:: ../_static/tensor_3d_sequence.svg
         :align: center
         :scale: 90%
         :alt: 3D Tensor

         3D Tensor

   Filter, Bias and Activation
     Unlike a tensor, the order of a filter, bias, and activation is flexible according to specific operations.

     For more details, please refer to :project_file:`include/typedef/dl_constant.hpp` or API documentation.
