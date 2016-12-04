using Encog.Engine.Network.Activation;
using Encog.ML.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace XOR_Demo
{
    class Program
    {
        static void Main(string[] args)
        {
			/*Inputs*/
            double[][] XOR_Input =
                {
                    new[] {0.0,0.0},
                    new[] {1.0,0.0},
                    new[] {0.0,1.0},
                    new[] {1.0,1.0}
                };

			/*Ideal Outputs*/
            double[][] XOR_Ideal =
                {
                    new[] {0.0},
                    new[] {1.0},
                    new[] {1.0},
                    new[] {0.0}
                };

			/* Pass inputs and ideal outputs */
            var trainingSet = new BasicMLDataSet(XOR_Input, XOR_Ideal);


            BasicNetwork network = CreateNetwork(); /* Creation of NN	*/

            var train = new ResilientPropagation(network, trainingSet); //created a training object
			/*	Training Process	*/
            int epoch = 1;
            do
            {

                train.Iteration();
                epoch++;
                Console.WriteLine("Iteration No :{0}, Error Viki: {1}", epoch, train.Error);

            } while (train.Error > 0.001); //loop until the error is less than a particular epoch


            foreach (var item in trainingSet) //Each item represents an IMLData Pair
            {
				var output = network.Compute(item.Input); //computes the predicted value from the NN model. output is an array of double
                Console.WriteLine("Input : {0}, {1} Ideal : {2} Actual : {3}", item.Input[0], item.Input[1], item.Ideal[0], output[0]);
			}


            Console.WriteLine("press any key to exit...");
            Console.ReadLine();
        }

		/*Creation of Neural Network */
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
    }
}
