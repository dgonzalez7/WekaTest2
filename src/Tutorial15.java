import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instances;


public class Tutorial15 {

	public static void main(String[] args) throws Exception 
	{
		BufferedReader breader = new BufferedReader(new FileReader("data/iris.arff"));
		
		
		Instances train = new Instances(breader);
		train.setClassIndex(train.numAttributes()-1);
		
		breader.close();
		
		NaiveBayes nB = new NaiveBayes();
		// nB.buildClassifier(train);		
		Evaluation eval = new Evaluation(train);
		eval.crossValidateModel(nB, train, 10, new Random(1));
		
		System.out.println(eval.toSummaryString("\nResults\n******\n", true));
		System.out.println(eval.fMeasure(0) + " " + eval.precision(0) + " " + eval.recall(0));
		System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));
		System.out.println(eval.fMeasure(2) + " " + eval.precision(2) + " " + eval.recall(2));

		System.out.println(eval.correct() + " " + eval.pctCorrect());
		System.out.println(train.classIndex());
	}
}
