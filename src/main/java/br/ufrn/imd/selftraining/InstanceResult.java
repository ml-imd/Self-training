package br.ufrn.imd.selftraining;

import java.util.ArrayList;
import java.util.TreeMap;

import weka.core.Instance;

public class InstanceResult {

	private Instance instance;
	private ArrayList<Double> predictions;
	private TreeMap<Double, Integer> agreementsPerClass;
	private Double bestClass;
	private Integer bestAgreement;

	public InstanceResult(Instance instance) {
		this.instance = instance;
		this.predictions = new ArrayList<Double>();
		this.agreementsPerClass = new TreeMap<Double, Integer>();
		this.bestClass = -1.0;
		this.bestAgreement = 0;
	}

	public void addPrediction(Double prediction) {
		this.predictions.add(prediction);
		Integer count = agreementsPerClass.containsKey(prediction) ? agreementsPerClass.get(prediction) : 0;
		agreementsPerClass.put(prediction, count + 1);

		if (agreementsPerClass.get(prediction) >= bestAgreement) {
			this.bestAgreement = agreementsPerClass.get(prediction);
			this.bestClass = prediction;
		}
	}

	public Instance getInstance() {
		return instance;
	}

	public void setInstance(Instance instance) {
		this.instance = instance;
	}

	public ArrayList<Double> getPredictions() {
		return predictions;
	}

	public void setPredictions(ArrayList<Double> predictions) {
		this.predictions = predictions;
	}

	public TreeMap<Double, Integer> getAgreementsPerClass() {
		return agreementsPerClass;
	}

	public void setAgreementsPerClass(TreeMap<Double, Integer> agreementsPerClass) {
		this.agreementsPerClass = agreementsPerClass;
	}

	public Double getBestClass() {
		return bestClass;
	}

	public void setBestClass(Double bestClass) {
		this.bestClass = bestClass;
	}

	public Integer getBestAgreement() {
		return bestAgreement;
	}

	public void setBestAgreement(Integer bestAgreement) {
		this.bestAgreement = bestAgreement;
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
		sb.append(agreementsPerClass.toString());
		sb.append(";");
		sb.append(bestClass);
		sb.append(";");
		sb.append(bestAgreement);

		return sb.toString();
	}

}
