package br.ufrn.imd.selftraining.bc.wekabuilders;

import weka.classifiers.trees.J48;

public class J48WekaBuilder {

	public static J48 buildForWeka(br.ufrn.imd.selftraining.bc.J48 classifier) {
		J48 j48 = new J48();
		
		j48.setUnpruned(classifier.getU());
		j48.setCollapseTree(classifier.getO());
		j48.setMinNumObj(classifier.getM());
		j48.setConfidenceFactor((float) classifier.getC());
		j48.setSubtreeRaising(classifier.getS());
		j48.setUseLaplace(classifier.getA());
		j48.setBinarySplits(classifier.getB());
		j48.setUseMDLcorrection(classifier.getJ());
		
		return j48;
	}
	
}
