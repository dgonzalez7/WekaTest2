import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;


public class Tutorial16 {

	public static void main(String[] args) throws IOException, Exception 
	{
		Instances instances = new Instances(new BufferedReader(new FileReader("data/iris.arff")));
		instances.setClassIndex(instances.numAttributes()-1);

		double precision = 0; 
		double recall = 0; 
		double fmeasure = 0; 
		double error = 0;
		
		// size is instances per fold
		int size = instances.numInstances() / 10;
		int begin = 0;
		int end = size - 1;
		
		for (int i = 1; i <= 10; i++)
		{
			System.out.println("Iteration: " + i);
			Instances trainingInstances = new Instances(instances);
			instances.randomize(new Random(1));
			Instances testingInstances = new Instances(instances, begin, (end - begin));
			for (int j = 0; j < (end - begin); j++)
			{
				trainingInstances.delete(begin);
			}
			
			NaiveBayes tree = new NaiveBayes();
			tree.buildClassifier(trainingInstances);
			
			Evaluation evaluation = new Evaluation(testingInstances);
			evaluation.evaluateModel(tree, testingInstances);
			
			System.out.println("P: " + evaluation.precision(1));
			System.out.println("R: " + evaluation.recall(1));
			System.out.println("F: " + evaluation.fMeasure(1));
			System.out.println("E: " + evaluation.errorRate());
			
			precision += evaluation.precision(1);
			recall += evaluation.recall(1);
			fmeasure += evaluation.fMeasure(1);
			error += evaluation.errorRate();
			
			// Update
			begin = end + 1;
			end += size;
			if (i == 9)
			{
				end = instances.numInstances();
			}
		}
		System.out.println("Precision: " + precision/10.0);
		System.out.println("Recall: " + recall/10.0);
		System.out.println("fMeasure: " + fmeasure/10.0);
		System.out.println("Error: " + error/10.0);
	}

}
