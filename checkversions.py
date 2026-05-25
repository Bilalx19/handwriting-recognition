#only for checking versions of the dependencies
# https://discuss.pytorch.org/t/how-do-i-know-the-current-version-of-pytorch/6754

# everything unnecessary since: pip show ... works


import torch
print(f"torch Version used:{torch.__version__}")

import torchvision
print(f"torchvision Version used:{torchvision.__version__}")

import flwr
print(f"flwr Version used:{flwr.__version__}")

import numpy
print(f"numpy Version used:{numpy.__version__}")

import PIL
print(f"PIL Version used:{PIL.__version__}")

import streamlit
print(f"streamlit Version used:{streamlit.__version__}")

##import streamlit_drawable_canvas
##print(f"streamlit_drawable_canvas Version used:{streamlit_drawable_canvas.__version__}") # wont work so do: pip show streamlit-drawable-canvas

import plotly
print(f"plotly Version used:{plotly.__version__}")

import pandas
print(f"pandas Version used:{pandas.__version__}")

## flwr-datasets[vision] ## Probably not needed