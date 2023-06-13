.. _explanations:

##############################
Parameter Explanations
##############################


Understanding the Trade-off Between Fidelity and Responsiveness in Window Size Adjustment
-----------------------------------------------------------------------------------------

Choosing an optimal window size in a matching algorithm is a key decision that influences the trade-off between fidelity (the ability to maintain accurate tracking of identities) and responsiveness (the ability to quickly respond to identity swaps). This is particularly relevant in the context of tracking multiple individuals in a video, where identity swaps and other confusions can occur, such as tracking bees in a hive. We use the Kuhn-Munkres algorithm for assigning identities based on detected ArUco tags on the individuals. However, the ArUco tag identification is not always perfect and sometimes the tag might not be read correctly or not read at all due to occlusions or other issues. Therefore, the choice of window size can have substantial effects on the resulting assignments and their accuracy.

In our framework, we construct a cost matrix ``C_{i,j}(t)`` from the binary matrix of tag ID and SLEAP instance coincidences, ``I_{i,j}(t)`` for each frame ``t``. Each element of the cost matrix is computed as the negative summation of the coincidences over an overlapping window of size ``2w+1`` centered on the frame:

.. math::

    C_{i,j}(t) = - \sum^{t+w}_{k = t-w} I_{i,j}(k)

We assign IDs to each instance by finding the minimum of the cost function using the Kuhn-Munkres algorithm as implemented in SciPy, represented by the following equation where ``A`` is a permutation matrix:

.. math::

    \hat{A}_{\text{arg min}} = \sum_{i\in\text{Tracks}}\ \sum_{j\in\text{Tags}} C_{i,j} A_{i,j}

Now, let's consider three different window sizes and their impact on the matching process:

1. **Small window size (3 frames; w=1)**

With a small window size, the algorithm can respond quickly to identity swaps, but it may also lead to an increase in misidentifications due to short-term ambiguities in tag readings. The cost matrix obtained may look like:

.. code-block:: none

    C_{i,j}(t) = [[-2, -1,  0],
                  [0, -1, -2],
                  [ 0, -2, -1]]

Here, due to the small window size, the first instance is correctly assigned to the first tag, but the second instance is incorrectly assigned to the second tag and the third instance to the second tag. This results in a deviation from the ideal diagonal alignment and shows how small windows can lead to misassignments.

1. **Medium window size (41 frames; w=20)**

Increasing the window size helps to mitigate the impact of missed tag readings but also delays the algorithm's response to identity swaps. The cost matrix obtained may look like:

.. code-block:: none

    C_{i,j}(t) = [[-20, -3, -2],
                  [-3, -21, -2],
                  [-2, -3, -20]]

Despite the missed tag readings, the larger window size allows for correct assignments as the correct identifications outweigh the missed tag identifications over the window of frames.

3. **Large window size (101 frames; w=50)**

With a large window size, the algorithm shows high fidelity in identity assignments, even in the case of numerous missed tag readings. However, it can be slow to respond to rapid identity swaps. The cost matrix obtained may look like:

.. code-block:: none

    C_{i,j}(t) = [[-50, -3, -2],
                  [-3, -51, -1],
                  [-2, -3, -51]]

Despite a substantial number of missed tag readings, the large window size ensures correct assignments as the correct identifications far outweigh the missed identifications over the large window of frames. However, a rapid identity swap would be missed due to the delayed response of the algorithm. 

These examples illustrate the need to carefully choose the window size based on the specifics of your experimental design and the nature of your data.
