
Has two versions, vanilla torch, and pytorch lightning version (see lightning branch)

# To run
Create a virtual env, it will save you so much headache
pip install -r requirements.txt

run the train.py file and make sure it runs.

Once you've convinced yourself the code is sound, you can read the code and understand the parts.
# For pytorch lightninig

in config file, set device to "cpu" if you run into issues with the dataloader. This will still train with the gpu (all of them), but keep the dataloader in the cpu, which is fine
you will still see much faster training
