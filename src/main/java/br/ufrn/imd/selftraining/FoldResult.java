package br.ufrn.imd.selftraining;

public class FoldResult {

	private double accuracy;
	private double error;
	private double fMeasure;
	private double precision;
	private double recall;
	

	public double getAccuracy() {
		return accuracy;
	}

	public void setAccuracy(double accuracy) {
		this.accuracy = accuracy;
	}

	public double getfMeasure() {
		return fMeasure;
	}

	public void setfMeasure(double fMeasure) {
		this.fMeasure = fMeasure;
	}

	public double getRecall() {
		return recall;
	}

	public void setRecall(double recall) {
		this.recall = recall;
	}

	public double getError() {
		return error;
	}

	public void setError(double error) {
		this.error = error;
	}

	public double getPrecision() {
		return precision;
	}

	public void setPrecision(double precision) {
		this.precision = precision;
	}
	
	public String onlyValuesToString() {
		
		StringBuilder sb = new StringBuilder();
		sb.append(formatValue(accuracy) + "\t\t");
		sb.append(formatValue(error) + "\t\t");
		sb.append(formatValue(fMeasure) + "\t\t");
		sb.append(formatValue(precision) + "\t\t");
		sb.append(formatValue(recall) + "\t\t");
		
		return sb.toString();
	}

	private String formatValue(Double value) {
		String s;
		if(value < 100) {
			s = String.format ("%.4f", value);
		}
		else {
			s = String.format ("%.3f", value);
		}
		return s;
	}
	
}
