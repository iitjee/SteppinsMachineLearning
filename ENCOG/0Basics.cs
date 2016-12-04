/*
- ENCOG is an advanced machine learning framework
-  Machine learning algorithms such as Support Vector Machines, Artificial Neural Networks, Bayesian Networks, Hidden Markov Models, Genetic Programming and Genetic Algorithms are supported.
- Most Encog training algoritms are multi-threaded and scale well to multicore hardware.
- C, C++, Java, and .NET are supported


Outline:
- Data
- Network
- Training
- Evaluation
- XOR Problem

ENCOG uses IMLData inteface for ENCOG
*/

For simple case use BasicMLData class
eg: 
double[] p = new double[10]; //Use double or Array of double values for NNs
IMLData data = new BasicMLData(p);

If you've smaller training data: Use Jagged array to provide smaller training input or output data instances
  double[][] XOR_Input = { new[] {0.0. 0.0}, new[] {0.0. 0.0}, new[] {0.0. 0.0}, new[] {0.0. 0.0} };
  
To create Training Data set: Use BasicMLDataSet class
  var trainingSet = new BasicMLDataSet(XOR_Input, XOR_Ideal);
