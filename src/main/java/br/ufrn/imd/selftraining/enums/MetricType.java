package br.ufrn.imd.selftraining.enums;

public enum MetricType {

	ACCURACY("accuracy"),
	ERROR("error"),
	PRECISION("precicion"),
	RECALL("recall"),
	F_MEASURE("f-measure");

	private String info;

	MetricType(String info) {
		this.info = info;
	}

	public String getInfo() {
		return this.info;
	}
	
}
