# NNVis
#### *Simple utils for neural network visualization*
---
Currently only layer activation distribution visualization functionality is available. 
### API
```python
# create your model
model = Model()

# put it in eval mode
model.eval()

# initialize visualizer object
# Violin or Ridge
nvis = NVIS("Violin", ["InputHook"])

# register hooks for your model
nvis(model)

# make a forward-pass for your model
model(data)

# export visualized data
nvis.export_pdf()
```
*The visualization will be located under vis/pdfs directory*