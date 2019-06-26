# Cyber Data Analytics Assignment 3: Data streams

## Requirements
- Python 3.7.1 (You really need 3.6+ as we used f-strings, you can try if it runs on 3.6 but there are no guarantees)
- Install the required packages from `requirements.txt` with `pip install -r requirements.txt`
- Download **unidirectional** netflows from 
[Scenario 1](https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-44/capture20110812.pcap.netflow.labeled) 
and [Scenario 10](https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-51/capture20110818.pcap.netflow.labeled)
from CTU
- Move the dataset to the `data/` directory with their original names

## Assignments
Below is a small overview of which python file belongs to which assignment.

**Sampling task**: implementation in `reservoir.py`, and tested in `compare.py`  
**Sketching task**: implementation in `sketch.py`, and tested in `compare.py`  
**Flow data discretization**: `flow_discretization.py`, the main method of the file computes the necessary plots  
**Botnet profiling**: `markovchain.py`, the main method of the file runs the profiling and creates adverserial examples  
**Flow classification** `classify.py`, the main method creates the necessary data and performs classification on the flows. 

## Notes
- **Check if the data_path variable is set correctly in each py file**.
- We used the interactive IPython kernel to interactively run code blocks indicated with `#%%`. If you also use Intelij, 
check out the [Scientific Mode](https://www.jetbrains.com/help/idea/matplotlib-tutorial.html) tutorial! A normal Python 
kernel ignores these statements and will run all code at once. Please check if the `data_path` variables is correct 
in those cases as the starting path of the kernel might be different. 

