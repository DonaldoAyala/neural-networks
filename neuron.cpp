#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;

/*
    Classs for modeling neural networks.
    Current version only allows a single layer.
*/
class NeuralNetwork 
{
    private:
        // weights[i][j] denote weight for input j of neuron i
        vector<vector<float>> weights;
        vector<float> bias;
        int neurons;
        int inputs;

        float hardlim(float n) 
        {
            return (n < 0) ? 0 : 1;
        }

    public:
        NeuralNetwork(int neurons, int inputs) 
        {
            this -> neurons = neurons;
            this -> inputs = inputs;
            weights.resize(neurons, vector<float>(inputs, 0.0f));
            bias.resize(neurons);
        }

        void printWeights() 
        {
            for (auto neuron: weights)
            {
                for (auto weight : neuron)
                {
                    cout << weight << " ";
                }
                cout << endl;
            }
        }

        NeuralNetwork(vector<vector<float>> weights, vector<float> bias) 
        {
            this -> weights = weights;
            this -> bias = bias;
            neurons = weights.size();
            inputs = weights.size() > 0 ? weights[0].size() : 0;
        }

        vector<float> process(vector<float> input) 
        {
            vector<float> output(neurons, 0);

            for (int i = 0; i < neurons; i++)
            {
                for (int j = 0; j < inputs; j++)
                {
                    output[i] += weights[i][j] + bias[i];
                }
                output[i] = hardlim(output[i]);
            }

            return output;
        }

        void setWeight(int neuron, int input, float value) 
        {
            weights[neuron][input] = value;
        }

        void setBias(int neuron, float bias) 
        {
            this -> bias[neuron] = bias;
        }

        float getWeight(int neuron, int input) 
        {
            return weights[neuron][input];
        }

        float getBias(int neuron) 
        {
            return bias[neuron];
        }

        int numInputs() 
        {
            return this -> inputs;
        }

        int numNeurons() 
        {
            return this -> neurons;
        }
};

class Trainer
{

    public:

        void train(NeuralNetwork* neuralNetwork, int iterations, float learningRate, vector<pair<vector<float>, float>> trainingSet)
        {
            while (iterations--)
            {
                neuralNetwork -> printWeights();
                vector<float> output;
                for (pair<vector<float>, float> example: trainingSet) 
                {
                    vector<float> input = example.first;
                    float expectedOutput = example.second;

                    output = neuralNetwork -> process(input);

                    for (int o = 0; o < neuralNetwork -> numNeurons(); o++)  
                    {
                        for (int i = 0; i < neuralNetwork -> numInputs(); i++) 
                        {
                            float newWeight = neuralNetwork -> getWeight(o, i) + learningRate * (expectedOutput - output[o]) * input[i];
                            neuralNetwork -> setWeight(o, i, newWeight);
                        }
                    }
                }
            }
        }
};

int main () {
    Trainer* trainer = new Trainer();
    NeuralNetwork* andNN = new NeuralNetwork(1, 2);
    vector<pair<vector<float>, float>> andTrainingSet = 
    {
        {{0, 0}, 0},
        {{0, 1}, 0},
        {{1, 0}, 0},
        {{1, 1}, 1},
    };

    trainer -> train(andNN, 10, 0.01f, andTrainingSet);

    cout << "AND" << endl;
    for (auto test : andTrainingSet) 
    {
        auto input = test.first;
        auto result = andNN -> process(input);
        cout << result.size() << endl;
        cout << input[0] << " " << input[1] << " = " << result[0] << endl;
    }
    return 0;
}