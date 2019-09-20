package br.ufrn.imd.selftraining.bc.wekabuilders;

import weka.classifiers.lazy.IBk;

public class IbkWekaBuilder {

	public static IBk buildForWeka(br.ufrn.imd.selftraining.bc.Ibk classifier) {
		IBk ibk = new IBk();
		
		ibk.setMeanSquared(classifier.getE());
		ibk.setCrossValidate(classifier.getX());
		ibk.setKNN(classifier.getK());
		ibk.setWindowSize(classifier.getW());
				
		return ibk;
	}
}
