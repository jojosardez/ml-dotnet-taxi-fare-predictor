using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Threading.Tasks;

namespace TaxiFarePrediction
{
    class Program
    {
        const string _datapath = @".\Data\taxi-fare-train.csv";
        const string _testdatapath = @".\Data\taxi-fare-test.csv";
        const string _modelpath = @".\Data\Model.zip";

        static async Task Main(string[] args)
        {
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = await Train();
            Evaluate(model);
        }

        static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
        {
            // Create learning pipeline
            var pipeline = new LearningPipeline
            {
                // Load and transform data
                new TextLoader(_datapath).CreateFrom<TaxiTrip>(separator: ','),

                 // Labeling
                new ColumnCopier(("FareAmount", "Label")),

                // Feature engineering
                new CategoricalOneHotVectorizer("VendorId",
                    "RateCode",
                    "PaymentType"),

                 // Combine features in a single vector
                new ColumnConcatenator("Features",
                    "VendorId",
                    "RateCode",
                    "PassengerCount",
                    "TripDistance",
                    "PaymentType"),

                // Add learning algorithm
                new FastTreeRegressor()
            };

            // Train the model
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();

            // Save the model to a zip file
            await model.WriteAsync(_modelpath);

            return model;
        }

        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            // Load test data
            var testData = new TextLoader(_datapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ',');

            // Evaluate test data
            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);

            // Display regression evaluation metrics
            // Rms should be around 2.795276
            Console.WriteLine("Rms=" + metrics.Rms);
            Console.WriteLine("RSquared = " + metrics.RSquared);
        }
    }
}
