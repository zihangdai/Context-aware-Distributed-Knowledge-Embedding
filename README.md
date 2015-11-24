# Context-aware Distributed Knowledge Embedding

The train.lua now contains everything, including the whole model. Yes, it is still very rough.

To run the code, install the following libraries (using luarocks):
    - cutorch
    - nn
    - cunn
    - nngraph
    - logroll (not used for now, but will be used in the future)

For training:
	- th train.lua -batchSize B -negSize N -useGPU 0/1 
	- B & N should be chosen largely based on the (GPU) memory available

For testing:
	- th test.lua -modelFile path_to_model 