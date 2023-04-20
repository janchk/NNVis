# NNVis
#### *Simple utils for neural network visualization*
---
Currently weights distribution and activations distrubution are available. 
### API
```python
# create your model
model = Model()

# put it in eval mode
model.eval()

# initialize visualizer object
nvis = NVIS()

# set model to Nvis
nvis.set_model(model)

# visualise weights distributions
nvis.plot_weights_ditributions()

# visualize activations distributions
nvis.plot_activations_distributions((<input_shape>))

```
*The visualization will be located under vis/pdfs directory*