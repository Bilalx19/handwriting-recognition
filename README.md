# handwriting-recognition
Handwriting recognition system for scanned student forms from schools. Each client holds data from a different school, leading to diverse handwriting styles. Focuses on privacy‑preserving and federated learning approaches to avoid sharing raw handwritten samples while enabling accurate handwritten text recognition (HTR).
The models accuracy (simulation purposed, client-side) is 86,127% at the moment.


## How to run APP? (without Docker)

Path: ..\handwriting-recognition
command: python -m streamlit run UI/app.py

make sure you have these packages installed: 
pip install torch torchvision numpy pillow streamlit streamlit-drawable-canvas plotly pandas
(this was suggested by AI while I tried to run the command)

But going trough the files, those packages only are important to install:

torch
torchvision
flwr
numpy
PIL
streamlit
streamlit_drawable_canvas
plotly
pandas
flwr-datasets[vision] ## not needed for the app User (docker part)???

To check the versions of them all do:

pip show torch torchvision flwr numpy PIL streamlit streamlit_drawable_canvas plotly pandas