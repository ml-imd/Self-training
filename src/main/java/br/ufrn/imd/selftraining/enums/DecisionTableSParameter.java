package br.ufrn.imd.selftraining.enums;

public enum DecisionTableSParameter {
	
	SN5("weka.attributeSelection.BestFirst -D 1 -N 5"), 
	SN7("weka.attributeSelection.BestFirst -D 1 -N 7"),
	SN3("weka.attributeSelection.BestFirst -D 1 -N 3");

	private String info;

	DecisionTableSParameter(String info) {
		this.info = info;
	}

	public String getInfo() {
		return this.info;
	}
}
