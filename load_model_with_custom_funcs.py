
import pickle
from custom_transformers import imput_all, feature_engineer  # ensure names are imported into __main__
# Note: If your pipeline referenced FunctionTransformer(imput_all) and FunctionTransformer(feature_engineer),
# having these names importable will allow pickle.load to succeed.
with open("Travel Insurance ML.sav", "rb") as f:
    pipe = pickle.load(f)
print("Loaded pipeline steps:", list(pipe.named_steps.keys()))
try:
    model = pipe.named_steps.get('model', None)
    print("Model:", type(model))
except Exception as e:
    print("Info:", e)
