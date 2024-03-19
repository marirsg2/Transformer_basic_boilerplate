
Has two versions, vanilla torch, and pytorch lightning version (see lightning branch)

# To run
pip install -r requirements.txt

run the train.py file and make sure it runs.

Once you've convinced yourself the code is sound, you can read the code and understand the parts.
# For pytorch lightninig
in config file, set "cpu" if you run into issues with the dataloader. This will still train with the cpu, but keep the dataloader in the cpu, which is fine
you will still see much faster training
