// the contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt

#include <dlib/svm_threaded.h>
#include <dlib/mlp.h>
#include <iostream>
#include <vector>

#include <dlib/rand.h>

using namespace std;
using namespace dlib;

// the data will be 2-dimensional data.
typedef matrix<double, 2, 1> sample_type_eyebrows;
typedef matrix<double, 2, 1> sample_type_mouth;

// method headers.
void generate_data_eyebrows(std::vector<sample_type_eyebrows>& samples, std::vector<double>& labels);
void generate_data_mouth(std::vector<sample_type_mouth>& samples, std::vector<double>& labels);


// loadClassifierAndPredictEyebrows() takes the eyebrows' data of the newly captured frame,
// and predicts emotion for new examples.
double loadClassifierAndPredictEyebrows(std::vector<double> eyebrows) {
	sample_type_eyebrows sample_eyebrows;

	for (int j = 0; j < eyebrows.size(); j++) {
		sample_eyebrows(j) = eyebrows[j];
	}

	typedef matrix<double, 2, 1> sample_type;
	typedef radial_basis_kernel<sample_type> kernel_type;
	typedef decision_function<kernel_type> dec_funct_type;
    typedef normalized_function<dec_funct_type> funct_type;
	funct_type  krr_classifier;
	deserialize("eyebrows_classifier.dat") >> krr_classifier;
	
	return krr_classifier(sample_eyebrows);
}

// trainClassifierEyebrows() trains the machine learning algorithm using the eyebrows' data obtained
// in the csv file of the eyebrows parameters.
void trainClassifierEyebrows() {
	// This typedef declares a matrix with 2 rows and 1 column.  It will be the
    // object that contains each of our 2 dimensional samples.   (Note that if you wanted 
    // more than 2 features in this vector you can simply change the 2 to something else.
    // Or if you don't know how many features you want until runtime then you can put a 0
    // here and use the matrix.set_size() member function)
    typedef matrix<double, 2, 1> sample_type;

    // This is a typedef for the type of kernel we are going to use in this example.
    // In this case I have selected the radial basis kernel that can operate on our
    // 2D sample_type objects
    typedef radial_basis_kernel<sample_type> kernel_type;


    // Now we make objects to contain our samples and their respective labels.
    std::vector<sample_type> samples;
    std::vector<double> labels;
	generate_data_eyebrows(samples, labels);

    cout << "samples generated: " << samples.size() << endl;
    cout << "  number of +1 samples: " << sum(mat(labels) > 0) << endl;
    cout << "  number of -1 samples: " << sum(mat(labels) < 0) << endl;

    // Here we normalize all the samples by subtracting their mean and dividing by their standard deviation.
    // This is generally a good idea since it often heads off numerical stability problems and also 
    // prevents one large feature from smothering others.  Doing this doesn't matter much in this example
    // so I'm just doing this here so you can see an easy way to accomplish this with 
    // the library.  
    vector_normalizer<sample_type> normalizer;
    // let the normalizer learn the mean and standard deviation of the samples
    normalizer.train(samples);
    // now normalize each sample
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]); 


    // here we make an instance of the krr_trainer object that uses our kernel type.
    krr_trainer<kernel_type> trainer;

    // The krr_trainer has the ability to perform leave-one-out cross-validation.
    // It does this to automatically determine the regularization parameter.  Since
    // we are performing classification instead of regression we should be sure to
    // call use_classification_loss_for_loo_cv().  This function tells it to measure 
    // errors in terms of the number of classification mistakes instead of mean squared 
    // error between decision function output values and labels.  
    trainer.use_classification_loss_for_loo_cv();


    // From looking at the output of the above loop it turns out that a good value for 
    // gamma for this problem is 0.000625.  So that is what we will use.
    trainer.set_kernel(kernel_type(0.000625));
    typedef decision_function<kernel_type> dec_funct_type;
    typedef normalized_function<dec_funct_type> funct_type;


    // Here we are making an instance of the normalized_function object.  This object provides a convenient 
    // way to store the vector normalization information along with the decision function we are
    // going to learn.  
    funct_type learned_function;
    learned_function.normalizer = normalizer;  // save normalization information
    learned_function.function = trainer.train(samples, labels); // perform the actual training and save the results

    // print out the number of basis vectors in the resulting decision function
    cout << "\nnumber of basis vectors in our learned_function is " 
         << learned_function.function.basis_vectors.size() << endl;

    // Now let's try this decision_function on some samples we haven't seen before.
    // The decision function will return values >= 0 for samples it predicts
    // are in the +1 class and numbers < 0 for samples it predicts to be in the -1 class.
    sample_type sample;

	serialize("eyebrows_classifier.dat") << learned_function;

    // Now let's open that file back up and load the function object it contains.
    deserialize("eyebrows_classifier.dat") >> learned_function;

    sample(0) = 2.70563;
    sample(1) = 28689.7;
    cout << "This is a -1 class example, the classifier output is " << learned_function(sample) << endl;

    sample(0) = 4.89199;
    sample(1) = 19501;
    cout << "This is a +1 class example, the classifier output is " << learned_function(sample) << endl;


    // We can also train a decision function that reports a well conditioned probability 
    // instead of just a number > 0 for the +1 class and < 0 for the -1 class.  An example 
    // of doing that follows:
    typedef probabilistic_decision_function<kernel_type> probabilistic_funct_type;  
    typedef normalized_function<probabilistic_funct_type> pfunct_type;

    // The train_probabilistic_decision_function() is going to perform 3-fold cross-validation.
    // So it is important that the +1 and -1 samples be distributed uniformly across all the folds.
    // calling randomize_samples() will make sure that is the case.   
    randomize_samples(samples, labels);

	while (1);
}

// generate_data_eyebrows() takes all the samples in the csv file and passes them to the machine learning algorithm.
void generate_data_eyebrows(std::vector<sample_type_eyebrows>& samples,	std::vector<double>& labels)
{
	//const long num = 50;

	sample_type_eyebrows m;

	dlib::rand rnd;
	string line;

	// to be changed accordingly.
	ifstream myfile("C:/Users/Omar/Desktop/eyebrows.csv");
	std::vector<string> data;

	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			data.push_back(line);
		}
	}
	std::vector<string> tmp;

	for (int i = 0; i < data.size(); i++) {
		string value;
		string new_dataset = data[i];
		int start_index = 0;
		int index = start_index;

		while (index < new_dataset.size()) {
			if (new_dataset[index] == ',') {
				value = new_dataset.substr(start_index, index - start_index);
				tmp.push_back(value);
				start_index = index + 2;
				index = start_index;
			}
			else {
				index++;
			}
		}
		value = new_dataset.substr(start_index, index - start_index);
		tmp.push_back(value);
	}

	for (int g = 0; g < tmp.size(); g += 3) {
		m(0) = atof(tmp[g].c_str());
		m(1) = atof(tmp[g + 1].c_str());
		samples.push_back(m);
		if (tmp[g + 2] == "ns") {
			// -1 --> non-surprise
			labels.push_back(-1);
		}
		else {
			// 1 --> surprise
			labels.push_back(1);
		}
	}
}

// loadClassifierAndPredictEyebrows() takes the mouth's data of the newly captured frame,
// and predicts emotion for new examples.
double loadClassifierAndPredictMouth(std::vector<double> mouth) {
	sample_type_mouth sample_mouth;

	for (int j = 0; j < mouth.size(); j++) {
		sample_mouth(j) = mouth[j];
	}

	typedef matrix<double, 2, 1> sample_type;
	typedef radial_basis_kernel<sample_type> kernel_type;
	typedef decision_function<kernel_type> dec_funct_type;
    typedef normalized_function<dec_funct_type> funct_type;
	funct_type  krr_classifier;
	deserialize("saved_function.dat") >> krr_classifier;

	return krr_classifier(sample_mouth);
}

// trainClassifierEyebrows() trains the machine learning algorithm using the mouth's data obtained
// in the csv file of the mouth parameters.
void trainClassifierMouth() {
	// This typedef declares a matrix with 2 rows and 1 column.  It will be the
    // object that contains each of our 2 dimensional samples.   (Note that if you wanted 
    // more than 2 features in this vector you can simply change the 2 to something else.
    // Or if you don't know how many features you want until runtime then you can put a 0
    // here and use the matrix.set_size() member function)
    typedef matrix<double, 2, 1> sample_type;

    // This is a typedef for the type of kernel we are going to use in this example.
    // In this case I have selected the radial basis kernel that can operate on our
    // 2D sample_type objects
    typedef radial_basis_kernel<sample_type> kernel_type;


    // Now we make objects to contain our samples and their respective labels.
    std::vector<sample_type> samples;
    std::vector<double> labels;
	generate_data_mouth(samples, labels);

    cout << "samples generated: " << samples.size() << endl;
    cout << "  number of +1 samples: " << sum(mat(labels) > 0) << endl;
    cout << "  number of -1 samples: " << sum(mat(labels) < 0) << endl;

    // Here we normalize all the samples by subtracting their mean and dividing by their standard deviation.
    // This is generally a good idea since it often heads off numerical stability problems and also 
    // prevents one large feature from smothering others.  Doing this doesn't matter much in this example
    // so I'm just doing this here so you can see an easy way to accomplish this with 
    // the library.  
    vector_normalizer<sample_type> normalizer;
    // let the normalizer learn the mean and standard deviation of the samples
    normalizer.train(samples);
    // now normalize each sample
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]); 


    // here we make an instance of the krr_trainer object that uses our kernel type.
    krr_trainer<kernel_type> trainer;

    // The krr_trainer has the ability to perform leave-one-out cross-validation.
    // It does this to automatically determine the regularization parameter.  Since
    // we are performing classification instead of regression we should be sure to
    // call use_classification_loss_for_loo_cv().  This function tells it to measure 
    // errors in terms of the number of classification mistakes instead of mean squared 
    // error between decision function output values and labels.  
    trainer.use_classification_loss_for_loo_cv();


    // From looking at the output of the above loop it turns out that a good value for 
    // gamma for this problem is 0.000625.  So that is what we will use.
    trainer.set_kernel(kernel_type(0.000625));
    typedef decision_function<kernel_type> dec_funct_type;
    typedef normalized_function<dec_funct_type> funct_type;


    // Here we are making an instance of the normalized_function object.  This object provides a convenient 
    // way to store the vector normalization information along with the decision function we are
    // going to learn.  
    funct_type learned_function;
    learned_function.normalizer = normalizer;  // save normalization information
    learned_function.function = trainer.train(samples, labels); // perform the actual training and save the results

    // print out the number of basis vectors in the resulting decision function
    cout << "\nnumber of basis vectors in our learned_function is " 
         << learned_function.function.basis_vectors.size() << endl;

    // Now let's try this decision_function on some samples we haven't seen before.
    // The decision function will return values >= 0 for samples it predicts
    // are in the +1 class and numbers < 0 for samples it predicts to be in the -1 class.
    sample_type sample;

	serialize("saved_function.dat") << learned_function;

    // Now let's open that file back up and load the function object it contains.
    deserialize("saved_function.dat") >> learned_function;

    sample(0) = 0.159236;
    sample(1) = 0.0136922;
    cout << "This is a +1 class example, the classifier output is " << learned_function(sample) << endl;

    sample(0) = 0.177122;
    sample(1) = 0.0143789;
    cout << "This is a +1 class example, the classifier output is " << learned_function(sample) << endl;

    sample(0) = 0.0863309;
    sample(1) = 0.00590032;
    cout << "This is a -1 class example, the classifier output is " << learned_function(sample) << endl;

    sample(0) = 0.0613718;
    sample(1) = 0.00476352;
    cout << "This is a -1 class example, the classifier output is " << learned_function(sample) << endl;

    // We can also train a decision function that reports a well conditioned probability 
    // instead of just a number > 0 for the +1 class and < 0 for the -1 class.  An example 
    // of doing that follows:
    typedef probabilistic_decision_function<kernel_type> probabilistic_funct_type;  
    typedef normalized_function<probabilistic_funct_type> pfunct_type;

    // The train_probabilistic_decision_function() is going to perform 3-fold cross-validation.
    // So it is important that the +1 and -1 samples be distributed uniformly across all the folds.
    // calling randomize_samples() will make sure that is the case.   
    randomize_samples(samples, labels);
	while (1);
}

// generate_data_eyebrows() takes all the samples in the csv file and passes them to the machine learning algorithm.
void generate_data_mouth(std::vector<sample_type_mouth>& samples, std::vector<double>& labels) {
	//const long num = 50;

	sample_type_mouth m;

	dlib::rand rnd;
	string line;

	// to be changed accordingly.
	ifstream myfile("C:/Users/Omar/Desktop/new_params.csv");
	std::vector<string> data;

	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			data.push_back(line);
		}
	}
	std::vector<string> tmp;

	for (int i = 0; i < data.size(); i++) {
		string value;
		string new_dataset = data[i];
		int start_index = 0;
		int index = start_index;

		while (index < new_dataset.size()) {
			if (new_dataset[index] == ',') {
				value = new_dataset.substr(start_index, index - start_index);
				tmp.push_back(value);
				start_index = index + 2;
				index = start_index;
			}
			else {
				index++;
			}
		}
		value = new_dataset.substr(start_index, index - start_index);
		tmp.push_back(value);
	}

	for (int g = 0; g < tmp.size(); g += 3) {
		m(0) = atof(tmp[g].c_str());
		m(1) = atof(tmp[g + 1].c_str());
		samples.push_back(m);
		if (tmp[g + 2] == "sad") {
			// -1 --> sadness
			labels.push_back(-1);
		}
		else {
			// 1 --> happiness
			labels.push_back(1);
		}
	}
}
