package br.ufrn.imd.selftraining.bc.wekabuilders;

import weka.classifiers.functions.SMO;

public class SmoWekaBuilder {

	public static SMO buildForWeka(br.ufrn.imd.selftraining.bc.Smo classifier) {
		
		SMO smo = new SMO();
		
		smo.setEpsilon(classifier.getP());
		smo.setC(classifier.getC());
		smo.setToleranceParameter(classifier.getL());
		smo.setRandomSeed((int) classifier.getW());
		smo.setNumFolds((int) classifier.getV());
		
		return smo;
	}
}
