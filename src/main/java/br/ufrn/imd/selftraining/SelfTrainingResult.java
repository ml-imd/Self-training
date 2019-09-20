package br.ufrn.imd.selftraining;

import java.util.ArrayList;

public class SelfTrainingResult {

	private int numFolds;
	private String datasetName;
	private String selfTrainingVersion;
	private ArrayList<FoldResult> results;
	private FoldResult averageResult;

	public SelfTrainingResult(int numFolds, String datasetName, String selfTrainingVersion) {
		this.numFolds = numFolds;
		this.datasetName = new String(datasetName);
		this.selfTrainingVersion = new String(selfTrainingVersion);
		this.results = new ArrayList<FoldResult>();
		this.averageResult = new FoldResult();
	}
	
	public void addFoldResult(FoldResult result) {
		
		if(results.size() < numFolds) {
			this.results.add(result);
			if(results.size() == numFolds) {
				calcAverageResult();
			}
		}
		else {
			System.out.println("result from this dataset is already full");
		}
	}
	
	public void calcAverageResult() {
		
		int num = results.size();
		
		double accuracy = 0.0;
		double error = 0.0;
		double fMeasure = 0.0;
		double precision = 0.0;
		double recall = 0.0;
				
		for(FoldResult fr: results) {
			accuracy += fr.getAccuracy();
			error += fr.getError();
			fMeasure += fr.getfMeasure();
			precision += fr.getPrecision();
			recall += fr.getRecall();
		}
		
		averageResult.setAccuracy(accuracy / num);
		averageResult.setError(error / num);
		averageResult.setfMeasure(fMeasure / num);
		averageResult.setPrecision(precision / num);
		averageResult.setRecall(recall / num);
	}
	
	public int getNumFolds() {
		return numFolds;
	}

	public void setNumFolds(int numFolds) {
		this.numFolds = numFolds;
	}

	public String getDatasetName() {
		return datasetName;
	}

	public void setDatasetName(String datasetName) {
		this.datasetName = datasetName;
	}

	public String getSelfTrainingVersion() {
		return selfTrainingVersion;
	}

	public void setSelfTrainingVersion(String selfTrainingVersion) {
		this.selfTrainingVersion = selfTrainingVersion;
	}

	public ArrayList<FoldResult> getResults() {
		return results;
	}

	public void setResults(ArrayList<FoldResult> results) {
		this.results = results;
	}

	public FoldResult getAverageResult() {
		return averageResult;
	}

	public void setAverageResult(FoldResult averageResult) {
		this.averageResult = averageResult;
	}

	@Override
	public String toString() {
		return "SelfTrainingResult [numFolds=" + numFolds + ", datasetName=" + datasetName + ", selfTrainingVersion="
				+ selfTrainingVersion + ", results=" + results + ", averageResult=" + averageResult + "]";
	}

	public void showResult() {
		System.out.println(buildResultString());
	}

	public String getResult() {
		return buildResultString();
	}
	
	private String buildResultString() {
		StringBuilder sb = new StringBuilder();
		sb.append("------------------------------------------------------------------------------------------------");
		sb.append("\n");
		sb.append("@DATASET: " + datasetName + "\n");
		sb.append("@Folds  : " + numFolds + "\n");
		sb.append("@STvers : " + selfTrainingVersion + "\n");
		sb.append("------------------------------------------------------------------------------------------------");
		sb.append("\n\t\t");
		sb.append("accura" + "\t\t");
		sb.append("error " + "\t\t");
		sb.append("fmeasu" + "\t\t");
		sb.append("precis" + "\t\t");
		sb.append("recall" + "\t\t\n");
		sb.append("------------------------------------------------------------------------------------------------");
		sb.append("\n");
		for(int i = 0; i < results.size();i++) {
			sb.append("fold" + (i+1) + ":\t\t");
			sb.append(results.get(i).onlyValuesToString() +"\n");
		}
		sb.append("\n");
		sb.append("------------------------------------------------------------------------------------------------");
		sb.append("\n");
		sb.append("AVERAG" + "\t\t");
		sb.append(averageResult.onlyValuesToString() +"\n");
		sb.append("------------------------------------------------------------------------------------------------");
		sb.append("\n\n\n");
		return sb.toString();
	}
	
}
