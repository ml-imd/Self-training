package br.ufrn.imd.selftraining.enums;

public enum IbkAParameter {
	
	EUCLIDIAN("weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""), 
	MANHATAM("weka.core.neighboursearch.LinearNNSearch -A \"weka.core.ManhattanDistance -R first-last\"");

	private String info;

	IbkAParameter(String info) {
		this.info = info;
	}

	public String getInfo() {
		return this.info;
	}
}
