package br.ufrn.imd.selftraining.results;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.TreeMap;

import weka.core.Instance;

public class InstanceResult{

	private Instance instance;
	private ArrayList<Double> predictions;
	private TreeMap<Double, Integer> agreementsPerClass;
	private Double bestClass;
	private Integer bestAgreement;
	private Double factor;

	private ArrayList<Double> dwsValues;
	private Double bestDwsValue;
	private Double bestDwsClass;
	
	public InstanceResult(Instance instance) {
		this.instance = instance;
		this.predictions = new ArrayList<Double>();
		this.agreementsPerClass = new TreeMap<Double, Integer>();
		this.bestClass = -1.0;
		this.bestAgreement = 0;
		this.factor = 0.0;
		
		this.dwsValues = new ArrayList<Double>();
		this.bestDwsValue = 0.0;
		this.bestDwsClass = -1.0;
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
	
	public void addDwsValues(ArrayList<Double> dwsValues) {
		for (int i = 0; i < dwsValues.size(); i++) {
			this.dwsValues.add(dwsValues.get(i));
			if (dwsValues.get(i) >= this.bestDwsValue) {
				this.bestDwsValue = dwsValues.get(i);
				this.bestDwsClass = new Double(i);
			}
		}
	}

	public int getBestClassIndex() {
		return this.bestClass.intValue();
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
		sb.append(agreementsPerClass.toString());
		sb.append(";");
		sb.append(bestClass);
		sb.append(";");
		sb.append(bestAgreement);
		sb.append(";");
		sb.append(factor);

		return sb.toString();
	}
	
	public static Comparator<InstanceResult> bestAgreementComparatorAsc = new Comparator<InstanceResult>() {

		public int compare(InstanceResult irs1, InstanceResult irs2) {
			double x = irs1.getBestAgreement() - irs2.getBestAgreement();
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

	public static Comparator<InstanceResult> bestAgreementComparatorDesc = new Comparator<InstanceResult>() {

		public int compare(InstanceResult irs1, InstanceResult irs2) {
			double x = irs2.getBestAgreement() - irs1.getBestAgreement();
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
	
	public static Comparator<InstanceResult> factorComparatorAsc = new Comparator<InstanceResult>() {

		public int compare(InstanceResult ir1, InstanceResult ir2) {
			double x = ir1.getFactor() - ir2.getFactor();
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

	public static Comparator<InstanceResult> factorComparatorDesc = new Comparator<InstanceResult>() {

		public int compare(InstanceResult ir1, InstanceResult ir2) {
			double x = ir2.getFactor() - ir1.getFactor();
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
	
	public static Comparator<InstanceResult> bestDwsValueComparatorAsc = new Comparator<InstanceResult>() {

		public int compare(InstanceResult irs1, InstanceResult irs2) {
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

	public static Comparator<InstanceResult> bestDwsValueComparatorDesc = new Comparator<InstanceResult>() {

		public int compare(InstanceResult irs1, InstanceResult irs2) {
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

	
	public void correctValues() {
		if(Double.isInfinite(bestDwsValue)) {
			bestDwsValue = Double.MAX_VALUE;
		}
	}
}
