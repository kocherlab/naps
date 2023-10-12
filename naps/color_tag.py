#!/usr/bin/env python
class ColorTagModel:

    def __init__(
        self,
        color_set: list,
        **kwargs,
    ):
        
        # Assign the color model & dict
        self._color_model_dict = {}
        self._colorModel = self._assignColorDict(color_set)

    @classmethod
    def withColorSet(cls, color_set, **kwargs):
        return cls(color_set, **kwargs)

    def detect(self, img):
        
        img_mean = np.mean(img) # more efficient to initialize here
        lowest_zscore = None
        lowest_zscore_color = None
        for color, values in self._color_model_dict.items():
            color_zscore = abs((img_mean - values[0]) / values[1])
            if lowest_zscore == None:
                
                lowest_zscore = color_zscore
                lowest_zscore_color = color
                
            elif color_zscore < lowest_zscore:
                
                lowest_zscore = color_zscore
                lowest_zscore_color = color                
        
        return [lowest_zscore_color]

    def _assignColorDict(self, color_set):

        # Define names of each possible ArUco tag OpenCV supports
        COLOR_DICT = {
            "BLACK": [42.2061, 7.1768], # mean, std
            "PURPLE": [180.6056, 20.9124],
        }
           
        for color in color_set:
            if color not in COLOR_DICT:
                raise Exception(f"Unable to assign color: {color} in {COLOR_DICT}") # {} prints keys in dict
                
            self._color_model_dict[color] = COLOR_DICT[color]
            


# In[20]:


import cv2
import pytest
import numpy as np

@pytest.mark.parametrize(
    "color_set",
    [
        "BLACK",
        "PURPLE"
    ],
)

def test_ColorTagModel_no_color_set_error():

    with pytest.raises(Exception) as e_info:
        test_model = ColorTagModel.withColorSet(color_set)
        test_model.detect([30,31,32])


# In[23]:


test_ColorTagModel_no_color_set_error()


# In[16]:


color_set = ["PURPLE", "BLACK"]

ColorTagModel.withColorSet(color_set)

