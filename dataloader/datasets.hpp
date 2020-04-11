/**
 * @file datasets.hpp
 * @author Kartik Dutt
 * 
 * File containing details for every datasets.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_DATASETS_HPP
#define MODELS_DATASETS_HPP

#include <iostream>

/**
 * Structure used to provide details about the dataset.
 */
struct DatasetDetails
{
  std::string datasetName;
  std::string trainDownloadUrl;
  std::string testDownloadUrl;
  std::string trainHash;
  std::string testHash;
  bool loadCSV;
  std::string trainPath;
  std::string testPath;
  // The following parameters are for CSVs only.
  size_t startTrainingInputFeatures;
  size_t endTrainingInputFeatures;
  size_t startTrainingPredictionFeatures;
  size_t endTrainingPredictionFeatures;
  size_t startTestingInputFeatures;
  size_t endTestingInputFeatures;
  bool dropHeader;

  // Default constructor.
  DatasetDetails() {/* Nothing to do here. */}

  // Constructor for initializing object.
  DatasetDetails(const std::string& datasetName,
                 const std::string& trainDownloadUrl,
                 const std::string& testDownloadUrl,
                 const std::string& trainHash,
                 const std::string& testHash,
                 const bool loadCSV,
                 const std::string& trainPath,
                 const std::string& testPath) :
                 datasetName(datasetName),
                 trainDownloadUrl(trainDownloadUrl),
                 testDownloadUrl(testDownloadUrl),
                 trainHash(trainHash),
                 testHash(testHash),
                 loadCSV(loadCSV),
                 trainPath(trainPath),
                 testPath(testPath)
  {
    // Nothing to do here.
  }
};

class Datasets
{
 public:
  const static DatasetDetails MNIST()
  {
    DatasetDetails mnistDetails("mnist",
        "https://raw.githubusercontent.com/kartikdutt18/mlpack-models-weights-and-datasets/master/mnist-dataset/mnist_train.csv",
        "https://raw.githubusercontent.com/kartikdutt18/mlpack-models-weights-and-datasets/master/mnist-dataset/mnist_test.csv",
        "cc10cd2dcac4fa2b67b8e9b7c90019cb8669668ed11a1ea71d980418785b5b11",
        "1e4c6240156c2316c012a655ff268e8b7f37b4a4dabe49853ab3a60af0ed1bca",
        true,
        "./../data/mnist_train.csv",
        "./../data/mnist_test.csv");
    mnistDetails.startTestingInputFeatures = 0;
    mnistDetails.endTestingInputFeatures = -1;
    mnistDetails.startTrainingInputFeatures = 1;
    mnistDetails.endTrainingInputFeatures = -1;
    mnistDetails.startTrainingPredictionFeatures = 0;
    mnistDetails.endTrainingPredictionFeatures = 0;
    mnistDetails.dropHeader = true;
    return mnistDetails;
  }
};

#endif
