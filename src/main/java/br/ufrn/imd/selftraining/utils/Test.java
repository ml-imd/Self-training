package br.ufrn.imd.selftraining.utils;

import br.ufrn.imd.selftraining.core.Dataset;
import weka.core.EuclideanDistance;
import weka.core.Instance;

public class Test {

	
	public static void main(String[] args) throws Exception {
		Dataset d = new Dataset("src/main/resources/datasets/experiment2/"+"Car.arff");
		Instance[] centroids = Mathematics.centroidsOf(d.getInstances());
		
		for(int i = 0; i < centroids.length; i++) {
			System.out.println("centroid " + i +" = " + centroids[i].toString());
		}
		
		System.out.println(d.getInstances().toString());
		
		
		
		
	}
	
	
}
