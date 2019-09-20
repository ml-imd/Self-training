package br.ufrn.imd.selftraining.bc.wekabuilders;

import weka.classifiers.bayes.NaiveBayes;

public class NaiveBayesWekaBuilder {

	public static NaiveBayes buildForWeka(br.ufrn.imd.selftraining.bc.NaiveBayes classifier) {
		NaiveBayes nb = new NaiveBayes();
		
		nb.setUseKernelEstimator(classifier.getD());
		nb.setDisplayModelInOldFormat(classifier.getO());
		nb.setUseKernelEstimator(classifier.getK());
		
		return nb;
	}
}
