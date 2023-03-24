#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;

/*
    Classs for modeling a single perceptron.
*/
class Perceptron 
{
    private:
        // weights[i] denote weight for input i
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
    /*
        Vector space model: 
        [
            1 -> hard / 0 -> soft, 
            1 -> round / irregular -> 0,
            1 -> wrinkled / 0 -> smooth
            weight in kg
        ]
    */
    // Apples -> 1 vs Oranges -> 0
    vector<pair<vector<float>, float>> fruitsTrainingSet = 
    {
        {{1, 0, 0, 0.325f}, 1}, 
        {{0, 1, 1, 0.350f}, 0}
    };

    Trainer* trainer = new Trainer();
    Perceptron* fruitPerceptron = new Perceptron(3, 0);

    trainer -> train(fruitPerceptron, 10, 0.1f, fruitsTrainingSet);
    
    vector<pair<vector<float>, float>> testSet = 
    {
        {{1, 0, 0, 0.400f}, 1}, 
        {{0, 1, 1, 0.200f}, 0}
    };

    for (auto test : testSet) 
    {
        auto input = test.first;
        float result = fruitPerceptron -> process(input);
        cout << "real vs. predicted: " << test.second << " " << result << endl;
    }
    
    return 0;
}