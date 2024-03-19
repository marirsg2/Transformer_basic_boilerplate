
This is the branch for the pytorch lightning version (see lightning branch)

# To run
pip install -r requirements.txt or do it manually

run the train.py file and make sure it runs.

Once you've convinced yourself the code is sound, you can read the code and understand the parts.

in config file, set device to "cpu" if you run into issues with the dataloader. This will still train with the gpu (all of them), but keep the dataloader in the cpu, which is fine
you will still see much faster training
