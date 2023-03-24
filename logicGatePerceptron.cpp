#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;

/*
    Classs for modeling neural networks.
    Current version only allows a single layer.
*/
class Perceptron 
{
    private:
        // weights[i][j] denote weight for input j of neuron i
        vector<float> weights;
        float bias;
        int inputs;

        float hardlim(float n) 
        {
            return (n < 0) ? 0 : 1;
        }

    public:
        Perceptron(int inputs, float bias) 
        {
            this -> inputs = inputs;
            weights.resize(inputs);
            this -> bias = bias;
        }

        void print() 
        {
            cout << "Weights: ";
            for (auto x: weights)
            {
                cout << x << " ";
            }
            cout << " Bias: " << bias;
            cout << endl;
        }

        float process(vector<float> input)
        {
            float output = 0;

            for (int i = 0; i < inputs; i++)
            {
                output += weights[i] * input[i];
            }
            output += bias;
            output = hardlim(output);

            return output;
        }

        void setWeight(int input, float value) 
        {
            weights[input] = value;
        }

        void setBias(float bias) 
        {
            this -> bias = bias;
        }

        float getWeight(int input) 
        {
            return weights[input];
        }

        float getBias()
        {
            return bias;
        }

        int numInputs() 
        {
            return this -> inputs;
        }
};

class Trainer
{

    public:

        void train(Perceptron* perceptron, int iterations, float learningRate, vector<pair<vector<float>, float>> trainingSet)
        {
            while (iterations--)
            {
                // perceptron -> print();
                float output;
                for (pair<vector<float>, float> example: trainingSet) 
                {
                    vector<float> input = example.first;
                    float expectedOutput = example.second;

                    output = perceptron -> process(input);
                    float error = expectedOutput - output;

                    // Update weights
                    for (int i = 0; i < perceptron -> numInputs(); i++)
                    {
                        float newWeight = perceptron -> getWeight(i) + learningRate * (error) * input[i];
                        perceptron -> setWeight(i, newWeight);
                    }

                    // Update bias
                    perceptron -> setBias(perceptron -> getBias() + error * learningRate);

                }
            }
        }
};

int main () {
    vector<pair<vector<float>, float>> andTrainingSet = 
    {
        {{0, 0}, 0},
        {{0, 1}, 0},
        {{1, 0}, 0},
        {{1, 1}, 1},
    };
    vector<pair<vector<float>, float>> orTrainingSet = 
    {
        {{0, 0}, 0},
        {{0, 1}, 1},
        {{1, 0}, 1},
        {{1, 1}, 1},
    };
    vector<pair<vector<float>, float>> xorTrainingSet = 
    {
        {{0, 0}, 0},
        {{0, 1}, 1},
        {{1, 0}, 1},
        {{1, 1}, 0},
    };

    Trainer* trainer = new Trainer();
    Perceptron* andPerceptron = new Perceptron(2, 0);
    Perceptron* orPerceptron = new Perceptron(2, 0);
    Perceptron* xorPerceptron = new Perceptron(2, 0);

    trainer -> train(andPerceptron, 10, 0.1f, andTrainingSet);
    trainer -> train(orPerceptron, 10, 0.1f, orTrainingSet);
    trainer -> train(xorPerceptron, 10, 0.1f, xorTrainingSet);
    
    printf("Input \t\tAND \tOR\t XOR\n");
    for (auto test : andTrainingSet) 
    {
        auto input = test.first;
        float andResult = andPerceptron -> process(input);
        float orResult = orPerceptron -> process(input);
        float xorResult = xorPerceptron -> process(input);
        printf("%.1f  %.1f \t%.1f \t%.1f \t%.1f\n", input[0], input[1], andResult, orResult, xorResult); 
    }
    return 0;
}