package br.ufrn.imd.selftraining.results;

import java.util.ArrayList;
import java.util.Comparator;

import weka.core.Instance;

public class InstanceResultStandard {

	private Instance instance;
	private ArrayList<Double> confidences;
	private Double bestClass;
	private Double bestConfidence;
	private Double factor;
	
	private ArrayList<Double> dwsValues;
	private Double bestDwsValue;
	private Double bestDwsClass;
	
	public InstanceResultStandard(Instance instance) {
		this.instance = instance;
		this.confidences = new ArrayList<Double>();
		this.bestClass = -1.0;
		this.bestConfidence = 0.0;
		this.factor = 0.0;
		
		this.dwsValues = new ArrayList<Double>();
		this.bestDwsValue = 0.0;
		this.bestDwsClass = -1.0;
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
	
	public void addDwsValues(ArrayList<Double> dwsValues) {
		for (int i = 0; i < dwsValues.size(); i++) {
			this.dwsValues.add(dwsValues.get(i));
			if (dwsValues.get(i) >= this.bestDwsValue) {
				this.bestDwsValue = dwsValues.get(i);
				this.bestDwsClass = new Double(i);
			}
		}
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
	
	/**
	 * 
	 * @return This method return one string under csv rules, separated by ";" and
	 *         with all data recorded inside object at the moment of method's call.
	 *         this is different of "outputDataToCsv()" cause adds the factor at the end of line
	 * 
	 */
	public String outputDataToCsvWithDistanceFactor() {
		StringBuilder sb = new StringBuilder();
		sb.append(instance.toString());
		sb.append(";");
		sb.append(confidences.toString());
		sb.append(";");
		sb.append(bestClass);
		sb.append(";");
		sb.append(bestConfidence);
		sb.append(";");
		sb.append(factor);

		return sb.toString();
	}
	
	/**
	 * 
	 * @return This method return one string under csv rules, separated by ";" and
	 *         with all data recorded inside object at the moment of method's call.
	 *         this is different of "outputDataToCsv()" cause adds the factor at the end of line
	 * 
	 */
	public String outputDataToCsvWithDwsValues() {
		StringBuilder sb = new StringBuilder();
		sb.append(instance.toString());
		sb.append(";");
		sb.append(confidences.toString());
		sb.append(";");
		sb.append(bestClass);
		sb.append(";");
		sb.append(bestConfidence);
		sb.append(";");
		sb.append(dwsValues.toString());

		return sb.toString();
	}

	public static Comparator<InstanceResultStandard> bestConfidenceComparatorAsc = new Comparator<InstanceResultStandard>() {

		public int compare(InstanceResultStandard irs1, InstanceResultStandard irs2) {
			double x = irs1.getBestConfidence() - irs2.getBestConfidence();
			if(x > 0) {
				return 1;
			}
			else if(x == 0) {
				return 0;
			}
			else{
				return -1;
			}
		}
	};

	public static Comparator<InstanceResultStandard> bestConfidenceComparatorDesc = new Comparator<InstanceResultStandard>() {

		public int compare(InstanceResultStandard irs1, InstanceResultStandard irs2) {
			double x = irs2.getBestConfidence() - irs2.getBestConfidence();
			if(x > 0) {
				return 1;
			}
			else if(x == 0) {
				return 0;
			}
			else{
				return -1;
			}
		}
	};
	
	public static Comparator<InstanceResultStandard> factorComparatorAsc = new Comparator<InstanceResultStandard>() {

		public int compare(InstanceResultStandard irs1, InstanceResultStandard irs2) {
			double x = irs1.getFactor() - irs2.getFactor();
			if(x > 0) {
				return 1;
			}
			else if(x == 0) {
				return 0;
			}
			else{
				return -1;
			}
		}
	};

	public static Comparator<InstanceResultStandard> factorComparatorDesc = new Comparator<InstanceResultStandard>() {

		public int compare(InstanceResultStandard irs1, InstanceResultStandard irs2) {
			double x = irs2.getFactor() - irs1.getFactor();
			if(x > 0) {
				return 1;
			}
			else if(x == 0) {
				return 0;
			}
			else{
				return -1;
			}
		}
	};
	
	public static Comparator<InstanceResultStandard> bestDwsValueComparatorAsc = new Comparator<InstanceResultStandard>() {

		public int compare(InstanceResultStandard irs1, InstanceResultStandard irs2) {
			double x = irs1.getBestDwsValue() - irs2.getBestDwsValue();
			if(x > 0) {
				return 1;
			}
			else if(x == 0) {
				return 0;
			}
			else{
				return -1;
			}
		}
	};

	public static Comparator<InstanceResultStandard> bestDwsValueComparatorDesc = new Comparator<InstanceResultStandard>() {

		public int compare(InstanceResultStandard irs1, InstanceResultStandard irs2) {
			double x = irs2.getBestDwsValue() - irs1.getBestDwsValue();
			if(x > 0) {
				return 1;
			}
			else if(x == 0) {
				return 0;
			}
			else{
				return -1;
			}
		}
	};
	
	public int getBestClassIndex() {
		return this.bestClass.intValue();
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

	public Double getFactor() {
		return factor;
	}

	public void setFactor(Double factor) {
		this.factor = factor;
	}

	public ArrayList<Double> getDwsValues() {
		return dwsValues;
	}

	public void setDwsValues(ArrayList<Double> dwsValues) {
		this.dwsValues = dwsValues;
	}

	public Double getBestDwsValue() {
		return bestDwsValue;
	}

	public void setBestDwsValue(Double bestDwsValue) {
		this.bestDwsValue = bestDwsValue;
	}

	public Double getBestDwsClass() {
		return bestDwsClass;
	}

	public void setBestDwsClass(Double bestDwsClass) {
		this.bestDwsClass = bestDwsClass;
	}

}
