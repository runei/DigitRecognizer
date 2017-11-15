#include <iostream>
#include <armadillo>
#include <string>
#include <vector>
#include <random>

#define N_OUTPUT 10

double sigmoid(const double);
double sigmoidPrime(const double);
void printImg(const arma::rowvec &);

struct Train
{
	arma::rowvec image;
	int output_number;
	arma::vec output;

	Train() : output(N_OUTPUT, arma::fill::zeros) {};
	void setOutput(const int n);
};

void Train::setOutput(const int n)
{
	output_number = n;
	output[n] = 1;
}

struct Network
{
	arma::vec layers_sizes;
	std::vector<arma::mat> weights;
	std::vector<arma::rowvec> biases;
	std::vector<arma::mat> activations;
	std::vector<arma::mat> zs;

	Network(const arma::vec &);
	arma::mat forward(const arma::rowvec &);
	std::tuple<std::vector<arma::mat>, std::vector<arma::mat>> backpropagation(const arma::mat &, const arma::mat &);
	void update(const std::vector<Train> &, const double, const double);
	void gradientDescendent(std::vector<Train> &, const int, const double, const double, const double);
	int evaluate(const std::vector<Train> &);
	void weightInitializer(const arma::vec &);
	void oldWeightInitializer(const arma::vec &);
};

Network::Network(const arma::vec & _sizes) : layers_sizes { _sizes }, weights(layers_sizes.size() - 1), biases(layers_sizes.size() - 1), activations(weights.size()), zs(weights.size())
{
	weightInitializer(_sizes);
};

void Network::weightInitializer(const arma::vec & _sizes)
{
	for (unsigned int i = 0; i < _sizes.size() - 1; ++i)
	{
		weights[i] = arma::randn<arma::mat>(_sizes[i], _sizes[i + 1]) / std::sqrt(_sizes[i]);
		biases[i] = arma::randn<arma::rowvec>(_sizes[i + 1]);
	}
}

void Network::oldWeightInitializer(const arma::vec & _sizes)
{
	for (unsigned int i = 0; i < _sizes.size() - 1; ++i)
	{
		weights[i] = arma::randn<arma::mat>(_sizes[i], _sizes[i + 1]);
		biases[i] = arma::randn<arma::rowvec>(_sizes[i + 1]);
	}
}

arma::mat Network::forward(const arma::rowvec & input)
{
	zs[0] = (input * weights[0]) + biases[0];
	activations[0] = zs[0];
	activations[0].transform(sigmoid);
	for (unsigned int i = 1; i < weights.size(); ++i)
	{
		zs[i] = (activations[i - 1] * weights[i]) + biases[i];
		activations[i] = zs[i];
		activations[i].transform(sigmoid);
	}
	return activations.back();
}

std::tuple<std::vector<arma::mat>, std::vector<arma::mat>> Network::backpropagation(const arma::mat & X, const arma::mat & y)
{
	arma::mat yHat { forward(X) };
	arma::mat error { yHat - y };
	std::vector<arma::mat> prime_zs(zs.size()), dJdW(zs.size()), dJdB(zs.size());
	arma::mat delta;

	int last_index = prime_zs.size() - 1;
	for (int i = last_index; i >= 0; --i)
	{
		prime_zs[i] = zs[i];
		prime_zs[i].transform(sigmoidPrime);
		if (i == last_index)
		{
			delta = error;// % prime_zs[i];
		}
		else
		{
			delta = (delta * weights[i + 1].t()) % prime_zs[i];
		}
		dJdB[i] = delta;
		if (i == 0)
		{
			dJdW[i] = X.t() * delta;
		}
		else
		{
			dJdW[i] = activations[i - 1].t() * delta;
		}
	}

	return std::make_tuple(dJdW, dJdB);
}

void Network::update(const std::vector<Train> & train_ds, const double learning_rate, const double regularization)
{
	std::vector<arma::mat> delta_w(weights);
	std::vector<arma::rowvec> delta_b(biases);
	for (unsigned int i = 0; i < delta_w.size(); ++i)
	{
		delta_w[i].zeros();
		delta_b[i].zeros();
	}
	double lr = learning_rate / train_ds.size();
	for (const Train & t : train_ds)
	{
		std::vector<arma::mat> dJdW, dJdB;//){backpropagation(t.image, t.output.t())};
		std::tie(dJdW, dJdB) = backpropagation(t.image, t.output.t());
		// dJdW = backpropagation(t.image, t.output.t());
		for (unsigned int i = 0; i < delta_w.size(); ++i)
		{
			delta_w[i] = delta_w[i] + dJdW[i];
			delta_b[i] = delta_b[i] + dJdB[i];
		}
		for (unsigned int i = 0; i < weights.size(); ++i)
		{
			weights[i] = (1 - learning_rate * (regularization / train_ds.size())) * weights[i] - (lr * delta_w[i]);
			biases[i] = biases[i] - (lr * delta_b[i]);
		}
	}
}

void Network::gradientDescendent(std::vector<Train> & dataset, const int attemps, const double learning_rate, const double perc_test, const double regularization)
{
	for (int i = 0; i < attemps; ++i)
	{
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(dataset.begin(), dataset.end(), g);
		std::vector<Train> train, test;
		int test_start = dataset.size() * perc_test;
		train.assign(dataset.begin(), dataset.end() - test_start);
		test.assign(dataset.end() - test_start, dataset.end());

		update(train, learning_rate, regularization);
		int correct_predits { evaluate(test) };

		double perc_correct = (double) correct_predits / (double) test.size() * 100.0;
		std::cout << "    " << correct_predits << " \\ " << test.size() << " = " << perc_correct << "%\n";
	}
}

int Network::evaluate(const std::vector<Train> & test)
{
	int result { 0 };
	for (const Train & t : test)
	{
		arma::mat predict { forward(t.image) };
		int i = 0;
		while (predict[i] != predict.max())
		{
			++i;
		}
		result += i == t.output_number;
	}
	return result;
}


arma::mat readCSV(const std::string & file_name)
{
	arma::mat result;
	result.load(file_name, arma::csv_ascii);
	return result;
}

std::vector<Train> getTrainDataSet(const std::string & file_name)
{
	arma::mat csv { readCSV(file_name) };
	std::vector<Train> result(csv.n_rows, Train());
	for (unsigned int i = 0; i < result.size(); ++i)
	{
		result[i].image = csv.submat(i, 1, i, csv.n_cols - 1);
		result[i].image = result[i].image / 255;
		result[i].setOutput(csv(i, 0));
	}
	return result;
}

double sigmoid(const double n)
{
	if (n < 0)
	{
		return 1.0 - 1.0 / (1.0 + std::exp(n));
	}
	return 1.0 / (1.0 + std::exp(-n));
}

double sigmoidPrime(const double n)
{
	return sigmoid(n) * (1 - sigmoid(n));
}

void printImg(const arma::rowvec & img)
{
	for (int i = 0; i < 28; ++i)
	{
		for (int j = 0; j < 28; ++j)
		{
			std::cout << img[i * 28 + j] << " ";
		}
		std::cout << "\n";
	}
}

int main()
{
	std::vector<Train> train { getTrainDataSet("train.csv") };

// std::cout << train.size();return 0;
	// printImg(train[2].image);return 0;
	std::cout << "Corretos" << " \\ " << "Total" << " = " << "Perc acertos\n";

	arma::vec layers_sizes{ 784, 30, 10 };
	Network net(layers_sizes);

	// int i = 0;
//	auto x = net.backpropagation(train[i].image, train[i].output.t());

	// x.back().t().print();
	net.gradientDescendent(train, 50, 0.1, 0.25, 5.0);

	return 0;
}