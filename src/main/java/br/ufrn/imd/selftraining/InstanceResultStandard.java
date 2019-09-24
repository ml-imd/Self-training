package br.ufrn.imd.selftraining;

import java.util.ArrayList;
import java.util.Comparator;

import weka.core.Instance;

public class InstanceResultStandard implements Comparable<InstanceResultStandard>{

	private Instance instance;
	private ArrayList<Double> confidences;
	private Double bestClass;
	private Double bestConfidence;

	public InstanceResultStandard(Instance instance) {
		this.instance = instance;
		this.confidences = new ArrayList<Double>();
		this.bestClass = -1.0;
		this.bestConfidence = 0.0;
	}

	public void addConfidences(double[] predictions) {
		for (int i = 0; i < predictions.length; i++) {
			this.confidences.add(predictions[i]);
			if (predictions[i] >= this.bestConfidence) {
				this.bestConfidence = predictions[i];
				this.bestClass = new Double(i);
			}
		}
	}

	public Instance getInstance() {
		return instance;
	}

	public void setInstance(Instance instance) {
		this.instance = instance;
	}

	public ArrayList<Double> getConfidences() {
		return confidences;
	}

	public void setConfidences(ArrayList<Double> confidences) {
		this.confidences = confidences;
	}

	public Double getBestClass() {
		return bestClass;
	}

	public void setBestClass(Double bestClass) {
		this.bestClass = bestClass;
	}

	public Double getBestConfidence() {
		return bestConfidence;
	}

	public void setBestConfidence(Double bestConfidence) {
		this.bestConfidence = bestConfidence;
	}

	/**
	 * 
	 * @return This method return one string under csv rules, separated by ";" and
	 *         with all data recorded inside object at the moment of method's call.
	 * 
	 */
	public String outputDataToCsv() {
		StringBuilder sb = new StringBuilder();
		sb.append(instance.toString());
		sb.append(";");
		sb.append(confidences.toString());
		sb.append(";");
		sb.append(bestClass);
		sb.append(";");
		sb.append(bestConfidence);

		return sb.toString();
	}
	
	public int compareTo(InstanceResultStandard irs) {
		return (int) (this.bestConfidence - irs.getBestConfidence());
	}

	public static Comparator<InstanceResultStandard> bestConfidenceComparatorAsc = new Comparator<InstanceResultStandard>() {

		public int compare(InstanceResultStandard irs1, InstanceResultStandard irs2) {
			double x = irs1.getBestConfidence() - irs2.getBestConfidence();
			x = x * 100000;
			return (int)x;
		}
	};

	public static Comparator<InstanceResultStandard> bestConfidenceComparatorDesc = new Comparator<InstanceResultStandard>() {

		public int compare(InstanceResultStandard irs1, InstanceResultStandard irs2) {
			double x = irs1.getBestConfidence() - irs2.getBestConfidence();
			x = x * 100000;
			return (int)-x;
		}
	};
}
