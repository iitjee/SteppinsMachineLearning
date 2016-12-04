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

/*Setup: Just open .csproj file in Visual Studio and run. Tut says to downlaod some ENCOG dll file but worked fine without
it. Just try one :)
*/

For simple case use BasicMLData class
eg: 
double[] p = new double[10]; //Use double or Array of double values for NNs
IMLData data = new BasicMLData(p);

If you've smaller training data: Use Jagged array to provide smaller training input or output data instances
  double[][] XOR_Input = { new[] {0.0. 0.0}, new[] {0.0. 0.0}, new[] {0.0. 0.0}, new[] {0.0. 0.0} };
  
To create Training Data set: Use BasicMLDataSet class
  var trainingSet = new BasicMLDataSet(XOR_Input, XOR_Ideal);



/* Creating Neural Network  */
        private static BasicNetwork CreateNetwork()
        {

			//Create an object of 'BasicNetwork' Class
            var network = new BasicNetwork(); 

			//Use BasicLayer class to create input, hidden and output layers
			network.AddLayer(new BasicLayer(null, true, 2)); //This is input layer and we haven't used any activation function(=> null). Two neurons are used
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 2)); //First hidden layer with sigmoid activation function with bias
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, 1)); //Output layer with sigmoid activation with no bias
			//1stparam =  Activation function should always implement IActivationFunction Interface
			//Here, we've used sigmoid activation function 
			//2ndparam = should use bias parameter for the layer or not 
			//3rdparam = Number of neurons in that layer

            network.Structure.FinalizeStructure(); //Tells fw that all the layers have been added now 
            network.Reset();//Randomly initialize the weights 
            return network;

        }
        
        
 /* Training Process */
 In ENCOG, Training algos are classes implementing the IMLTrain Interface
    var train = new ResilientPropagation(network, trainingSet); //created a training object
 
            int epoch = 1;
            do
            {

                train.Iteration();
                epoch++;
                Console.WriteLine("Iteration No :{0}, Error: {1}", epoch, train.Error);

            } while (train.Error > 0.001); //loop until the error is less than a particular epoch
            
            
            
            
/*  Evaluation  */

    foreach (var item in trainingSet) //we pass each data set
            {
            var output = network.Compute(item.Input); //You can use compute method of trained network to calculate the output
            //Note that the input should be a double array or implement IMLDataInterface
            
            Console.WriteLine("Input : {0}, {1} Ideal : {2} Actual : {3}", item.Input[0], item.Input[1], item.Ideal[0], output[0]);
            }

//Complte problem solution @ https://github.com/iitjee/SteppinsMachineLearning/blob/master/ENCOG/1XORproblem.cs
