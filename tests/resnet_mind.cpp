/**
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack.hpp>
// #include <utils/utils.hpp>
#include <dataloader/dataloader.hpp>
// #include <models/resnet/resnet.hpp>

// using namespace mlpack;
using namespace mlpack::models;

void resnet_test(void) {

    DataLoader<> irisDataloader;

    const std::string datasetPath = "mnist";
    bool shuffleData = true;
    double ratioForTrainTestSplit = 0.75;
    bool isTrainingData = true;
    bool useFeatureScaling = true;
    bool dropHeader = false;

    // Starting column index for Training Features.
    size_t startInputFeatures = 0;
    // Ending column index for training Features.
    // We also support wrapped index i.e. -1 implies last column and so on.
    size_t endInputFeatures = -2;

    irisDataloader(datasetPath, isTrainingData, shuffleData, ratioForTrainTestSplit,
        useFeatureScaling, dropHeader, startInputFeatures, endInputFeatures);
}

int main(void)
{
  resnet_test();
}
