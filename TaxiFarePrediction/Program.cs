using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
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
    }
}
