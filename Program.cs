using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using System.IO;
using System.Collections;

namespace InferSandbox
{
    class Program
    {
        static ArrayList readData(string fileName)
        {
            ArrayList data = new ArrayList();
            StreamReader sr = new StreamReader(fileName);
            
            while (!sr.EndOfStream)
            {
                string[] tokens = sr.ReadLine().Split(",".ToCharArray());
                double[] dataLine = new double[tokens.Length];
                for(int i=0; i<tokens.Length; i++)
                {
                    dataLine[i]=double.Parse(tokens[i]);
                }
                data.Add(dataLine);
            }
            
            // arrange the data the other way
            ArrayList arrangedData=new ArrayList();
            for (int i = 0; i < ((double[])(data[0])).Length; i++)
            {
                double[] line = new double[data.Count];
                for (int j = 0; j < data.Count; j++)
                    line[j] = ((double[])(data[j]))[i];
                arrangedData.Add(line);
            }
            return arrangedData;
        }

        static void Main(string[] args)
        {
            // data
            ArrayList trainingData = readData(@"C:\Users\Jeremy\Documents\Visual Studio 2010\Projects\InferSandbox\train.txt");
            
            // Create target y
            VariableArray<double> y = Variable.Observed((double[])(trainingData[trainingData.Count-1])).Named("y");
            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(trainingData.Count),
                PositiveDefiniteMatrix.Identity(trainingData.Count))).Named("w");
            trainingData.RemoveAt(trainingData.Count - 1);
            BayesPointMachine(trainingData, w, y);

            InferenceEngine engine = new InferenceEngine();
            if (!(engine.Algorithm is GibbsSampling))
            {
                VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);
                Console.WriteLine("Dist over w=\n" + wPosterior);

                ArrayList testData = readData(@"C:\Users\Jeremy\Documents\Visual Studio 2010\Projects\InferSandbox\test.txt");

                VariableArray<double> ytest = Variable.Array<double>(new Range(((double[])(testData[0])).Length)).Named("ytest");
                BayesPointMachine(testData, Variable.Random(wPosterior).Named("w"), ytest);
                Console.WriteLine("output=\n" + engine.Infer(ytest));
            }
            else Console.WriteLine("This model has a non-conjugate factor, and therefore cannot use Gibbs sampling");

        }

        public static void BayesPointMachine(ArrayList trainingData, Variable<Vector> w, VariableArray<double> y)
        {
            // Create x vector, augmented by 1
            Range j = y.Range.Named("person");
            Vector[] xdata = new Vector[((double[])(trainingData[0])).Length];
            
            for (int i = 0; i < xdata.Length; i++) {
                double[] argumentList = new double[trainingData.Count+1];
                for (int k = 0; k < trainingData.Count; k++)
                    argumentList[k] = ((double[])(trainingData[k]))[i];
                argumentList[trainingData.Count] = 1;
                xdata[i] = Vector.FromArray(argumentList);
            }
            VariableArray<Vector> x = Variable.Observed(xdata, j).Named("x");

            // Bayes Point Machine
            double noise = 0.1;
            y[j] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[j]).Named("innerProduct"), noise);
        }

    }
}
