# GSO & EvoGSO

This code was written by the authors of:
<br>

**Gumbel-softmax-based Optimization: A Simple General Framework for Optimization Problems on Graphs**<br>
Yaoxin Li, Jing Liu, Guozheng Lin, Yueyuan Hou, Muyun Mou, Jiang Zhang<sup>\*</sup>(<sup>\*</sup>: Corresponding author) <br>
[Download PDF](https://arxiv.org/abs/2004.07300)<br>

<br>

### Abstract: 

In computer science, there exist a large number of optimization problems defined on graphs, that is to find a best node state configuration or a network structure such that the designed objective function is optimized under some constraints. However, these problems are notorious for their hardness to solve because most of them are NP-hard or NP-complete. Although traditional general methods such as simulated annealing (SA), genetic algorithms (GA) and so forth have been devised to these hard problems, their accuracy and time consumption are not satisfying in practice. In this work, we proposed a simple, fast, and general algorithm framework based on advanced automatic differentiation technique empowered by deep learning frameworks. By introducing Gumbel-softmax technique, we can optimize the objective function directly by gradient descent algorithm regardless of the discrete nature of variables. We also introduce evolution strategy to parallel version of our algorithm. We test our algorithm on three representative optimization problems on graph including modularity optimization from network science, Sherrington-Kirkpatrick (SK) model from statistical physics, maximum independent set (MIS) and minimum vertex cover (MVC) problem from combinatorial optimization on graph. High-quality solutions can be obtained with much less time consuming compared to traditional approaches.


### Requirements

- Python 3.6
- Pytorch 1.4

### Postscript
Our implementation is in PyTorch. The graphs used in the experiments are included as well. 
Since we haven't worked on integration of our code, the annotations may not be as clear and adequate as you expect. You may also find part of the code are in .py while the others are in Jupyter Notebook. And many of them may not be the exact same version of experiments in our paper since we've tried other attempts afterwards. 


Nevertheless, You are welcome to make possible improvements based on these code.
