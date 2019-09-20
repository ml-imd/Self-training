package br.ufrn.imd.selftraining.enums;

public enum ClassifierType {

	BASE_CLASSIFIER("baseClassifier"), 
	COMMITTEE("committee");

	private String info;

	ClassifierType(String info) {
		this.info = info;
	}

	public String getInfo() {
		return this.info;
	}
}
