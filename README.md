# Marine



[dataset](https://drive.google.com/file/d/1xr0q_Gj8d7LUxXTEx1517ll0sV63BCRO/view?usp=sharing)  

### Requirements

- Python 3
- numpy
- torch



### Usage

Please run `python Marine.py --help`.



### Input

- Edge File (all values should be integer type)
  > `NodeCount RelationCount`  
  > `U V R` (edge from node **U** to node **V** with relation **R**)  
  > ...

- Attrubute File (with alpha > 0.0)
  > `NodeCount Dimension`  
  > `x0 x1 x2 ...` (Node0)  
  > `x0 x1 x2 ...` (Node1)  
  > ...



### Output

- node.txt & rela.txt & link.txt
> `EntityCount Dimension`  
> `x0 x1 x2 ...` (Index0)  
> `x0 x1 x2 ...` (Index1)  
> ...



### Link Prediction Evaluation

Compile: `g++ -std=c++11 -fopenmp predion_eval.cpp -o eval`  
Usage: see `predion_eval.cpp` around Line 40  


